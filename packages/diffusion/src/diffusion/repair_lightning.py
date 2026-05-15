"""PyTorch Lightning training path for deterministic terrain repair."""

from __future__ import annotations

if __package__ is None:
    import sys
    from pathlib import Path

    _src = Path(__file__).resolve().parent.parent
    _src_s = str(_src)
    if _src_s not in sys.path:
        sys.path.insert(0, _src_s)

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import LitLogger
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=FutureWarning,
)

from diffusion.repair_data import TerrainRepairDataset
from diffusion.repair_model import TerrainRepairUNet
from diffusion.repair_training import (
    RepairLossWeights,
    RepairTrainingState,
    build_repair_checkpoint_meta,
    checkpoint_sibling,
    compute_repair_losses,
    configure_cuda_backend,
    evaluate_repair_cases,
    load_repair_checkpoint,
    resolve_training_export_dirs,
    save_repair_checkpoint,
)


class RepairLitLogger(LitLogger):
    """LitLogger without Lightning's ``log_graph`` :class:`UserWarning` (graphs are TensorBoard-only)."""

    @rank_zero_only
    def log_graph(self, model: Module, input_array: Tensor | None = None) -> None:
        return


def _resolve_num_workers(requested: int) -> int:
    """Map ``--num-workers``: ``-1`` → ``min(8, cpu_count - 1)``; ``0`` → main-process only."""
    if requested < 0:
        cpu = os.cpu_count() or 1
        return min(8, max(0, cpu - 1))
    return requested


def _format_repair_batch(batch: dict[str, torch.Tensor], channels_last: bool) -> dict[str, torch.Tensor]:
    if not channels_last:
        return batch
    return {
        k: v.contiguous(memory_format=torch.channels_last)
        if isinstance(v, torch.Tensor) and v.ndim == 4 and v.is_floating_point()
        else v
        for k, v in batch.items()
    }


def _precision_from_amp(amp: str) -> str:
    if amp == "off":
        return "32-true"
    if amp == "fp16":
        return "16-mixed"
    if amp == "bf16":
        return "bf16-mixed"
    # auto
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16-mixed"
    if torch.cuda.is_available():
        return "16-mixed"
    return "32-true"

class TerrainRepairLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_material_classes: int,
        learning_rate: float = 1e-4,
        channels_last: bool = False,
        weights: RepairLossWeights = RepairLossWeights(),
    ):
        super().__init__()
        self.save_hyperparameters({
            "num_material_classes": num_material_classes,
            "learning_rate": learning_rate,
            "channels_last": channels_last,
            "loss_weights": {
                "height": weights.height,
                "gradient": weights.gradient,
                "seam": weights.seam,
                "material": weights.material,
                "support": weights.support,
            },
        })
        self.model = TerrainRepairUNet(num_material_classes=num_material_classes)
        self.learning_rate = learning_rate
        self.channels_last = channels_last
        self.weights = weights
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

    def forward(
        self,
        known_height: torch.Tensor,
        prefill_height: torch.Tensor,
        mask: torch.Tensor,
        known_material: torch.Tensor,
        known_support: torch.Tensor,
        boundary_distance: torch.Tensor,
        prefill_gradients: torch.Tensor,
        prefill_laplacian: torch.Tensor,
    ):
        return self.model(
            known_height=known_height,
            prefill_height=prefill_height,
            mask=mask,
            known_material=known_material,
            known_support=known_support,
            boundary_distance=boundary_distance,
            prefill_gradients=prefill_gradients,
            prefill_laplacian=prefill_laplacian,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch = _format_repair_batch(batch, self.channels_last)
        losses = compute_repair_losses(self.model, batch, weights=self.weights)
        batch_size = batch["target_height"].shape[0]
        self.log("train/total_loss",    losses.total_loss,    on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/height_loss",   losses.height_loss,   on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/gradient_loss", losses.gradient_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/seam_loss",     losses.seam_loss,     on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/material_loss", losses.material_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/support_loss",  losses.support_loss,  on_step=True, on_epoch=True, batch_size=batch_size)
        return losses.total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer is not None else 100,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "train/total_loss"},
        }

class TerrainRepairDataModule(pl.LightningDataModule):
    def __init__(
        self,
        export_dirs: list[Path],
        tile_size: int,
        stride_chunks: int,
        mask_mode: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.export_dirs = export_dirs
        self.tile_size = tile_size
        self.stride_chunks = stride_chunks
        self.mask_mode = mask_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset: TerrainRepairDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.dataset is None:
            self.dataset = TerrainRepairDataset(
                self.export_dirs,
                tile_size=self.tile_size,
                stride_chunks=self.stride_chunks,
                mask_mode=self.mask_mode,
            )

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup("fit")
        assert self.dataset is not None
        loader_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(self.dataset, **loader_kwargs)

class RepairEpochCallback(pl.Callback):
    """Advance the dataset's mask epoch counter at the start of each training epoch."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: TerrainRepairLightningModule) -> None:
        dataset = getattr(trainer.datamodule, "dataset", None)
        if isinstance(dataset, TerrainRepairDataset):
            dataset.set_mask_epoch(trainer.current_epoch)


class RepairValidationCaseCallback(pl.Callback):
    """Run evaluation on held-out repair cases and log metrics via self.log."""

    def __init__(self, cases_dir: str | Path | None, validate_every: int, channels_last: bool):
        self.cases_dir = cases_dir
        self.validate_every = validate_every
        self.channels_last = channels_last
        self.best_score = float("inf")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: TerrainRepairLightningModule) -> None:
        if not trainer.is_global_zero or self.cases_dir is None or self.validate_every <= 0:
            return
        completed_epoch = trainer.current_epoch + 1
        if completed_epoch % self.validate_every != 0 and completed_epoch != trainer.max_epochs:
            return

        amp_dtype: torch.dtype | None
        if trainer.precision == "16-mixed":
            amp_dtype = torch.float16
        elif trainer.precision in ("bf16-mixed", "bf16"):
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = None

        AmpConfig = type("AmpConfig", (), {"enabled": amp_dtype is not None, "dtype": amp_dtype})
        metrics = evaluate_repair_cases(
            pl_module.model,
            self.cases_dir,
            device=pl_module.device,
            amp_config=AmpConfig(),
            channels_last=self.channels_last,
        )
        if metrics is None:
            return

        self.best_score = min(self.best_score, metrics.score)
        pl_module.log("val/score",             metrics.score,             sync_dist=False)
        pl_module.log("val/height_mae",        metrics.height_mae,        sync_dist=False)
        pl_module.log("val/seam_mae",          metrics.seam_mae,          sync_dist=False)
        pl_module.log("val/material_accuracy", metrics.material_accuracy,  sync_dist=False)
        pl_module.log("val/support_mse",       metrics.support_mse,       sync_dist=False)
        print(
            f"repair validation: score={metrics.score:.4f} height_mae={metrics.height_mae:.4f} "
            f"seam_mae={metrics.seam_mae:.4f} material_acc={metrics.material_accuracy:.4f} "
            f"support_mse={metrics.support_mse:.4f} cases={metrics.case_count}"
        )


class RepairCompatibleCheckpointCallback(pl.Callback):
    """Save repair.pt-compatible checkpoints alongside the Lightning .ckpt files.

    Also uploads the best checkpoint as a model artifact to LitLogger when
    ``log_model`` is enabled on the trainer's LitLogger instance.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        latest_checkpoint_path: str | Path,
        best_checkpoint_path: str | Path,
        args: argparse.Namespace,
        datamodule: TerrainRepairDataModule,
        validation_callback: RepairValidationCaseCallback,
        save_every: int,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.latest_checkpoint_path = Path(latest_checkpoint_path)
        self.best_checkpoint_path = Path(best_checkpoint_path)
        self.args = args
        self.datamodule = datamodule
        self.validation_callback = validation_callback
        self.save_every = save_every
        self.best_score = float("inf")

    def _find_lit_logger(self, trainer: pl.Trainer) -> LitLogger | None:
        for lg in trainer.loggers:
            if isinstance(lg, LitLogger):
                return lg
        return None

    def _save(self, path: Path, trainer: pl.Trainer, pl_module: TerrainRepairLightningModule, interrupted: bool) -> None:
        dataset = self.datamodule.dataset
        if dataset is None:
            return
        state = RepairTrainingState(
            completed_epochs=trainer.current_epoch + 1,
            global_step=trainer.global_step,
        )
        export_args = argparse.Namespace(**vars(self.args), best_score=self.validation_callback.best_score)
        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        meta = build_repair_checkpoint_meta(export_args, dataset, state, interrupted=interrupted)
        meta["trainer"] = "lightning"
        save_repair_checkpoint(path, pl_module.model, optimizer, meta=meta)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: TerrainRepairLightningModule) -> None:
        if not trainer.is_global_zero or self.save_every <= 0:
            return
        completed_epoch = trainer.current_epoch + 1
        if completed_epoch % self.save_every != 0 and completed_epoch != trainer.max_epochs:
            return

        self._save(self.checkpoint_path, trainer, pl_module, interrupted=False)
        if self.latest_checkpoint_path != self.checkpoint_path:
            self._save(self.latest_checkpoint_path, trainer, pl_module, interrupted=False)

        if self.validation_callback.best_score < self.best_score:
            self.best_score = self.validation_callback.best_score
            self._save(self.best_checkpoint_path, trainer, pl_module, interrupted=False)

            lit_logger = self._find_lit_logger(trainer)
            if lit_logger is not None:
                lit_logger.log_model_artifact(str(self.best_checkpoint_path))

    def on_exception(self, trainer: pl.Trainer, pl_module: TerrainRepairLightningModule, exception: BaseException) -> None:
        if trainer.is_global_zero:
            self._save(self.checkpoint_path, trainer, pl_module, interrupted=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train deterministic terrain repair with PyTorch Lightning.")

    parser.add_argument("--export-dir", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latest-checkpoint", default=None)
    parser.add_argument("--best-checkpoint", default=None)
    parser.add_argument("--resume", default=None, help="Load existing repair.pt weights before Lightning fit.")
    parser.add_argument("--ckpt-path", default=None, help="Resume a Lightning .ckpt training state.")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--stride-chunks", type=int, default=1)
    parser.add_argument("--mask-mode", default="terrain_mixed",
                        choices=["none", "rectangle", "strip", "blob", "mixed", "terrain_mixed"])

    parser.add_argument("--amp", default="auto", choices=["auto", "off", "fp16", "bf16"])
    parser.add_argument("--precision", default=None,
                        help="Lightning precision string (overrides --amp): 16-mixed, bf16-mixed, 32-true, …")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=-1,
                        help="DataLoader workers: -1 → min(8, cpu_count-1), 0 → main process only.")
    parser.add_argument("--matmul-precision", default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--tf32", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])

    parser.add_argument("--litlogger-root-dir", default=None,
                        help="Root directory for LitLogger local files (default: <lightning-root-dir>/litlogger).")
    parser.add_argument("--litlogger-name", default=None,
                        help="Experiment name shown in the Lightning AI dashboard (default: auto-generated).")
    parser.add_argument("--litlogger-teamspace", default=None,
                        help="Teamspace to attach this experiment to (format: username/teamspace or just teamspace).")
    parser.add_argument("--litlogger-metadata", action="append", default=None, metavar="KEY=VAL",
                        help="Extra key=value tags for the experiment (repeatable).")
    parser.add_argument("--litlogger-log-model", action="store_true",
                        help="Auto-upload Lightning .ckpt files as artifacts after each ModelCheckpoint save.")
    parser.add_argument("--no-litlogger-save-logs", action="store_true",
                        help="Disable terminal log capture.")

    parser.add_argument("--validation-cases-dir", default=None)
    parser.add_argument("--validate-every", type=int, default=1)

    parser.add_argument("--lightning-root-dir", default="./artifacts/lightning")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--limit-train-batches", default=None,
                        help="Fraction (0–1) or absolute count of batches per epoch.")
    parser.add_argument("--fast-dev-run", action="store_true")

    args = parser.parse_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)
    configure_cuda_backend(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        args.tf32,
        args.cudnn_benchmark,
    )

    export_dirs = resolve_training_export_dirs(args.export_dir)
    num_workers = _resolve_num_workers(args.num_workers)
    datamodule = TerrainRepairDataModule(
        export_dirs=export_dirs,
        tile_size=args.tile_size,
        stride_chunks=args.stride_chunks,
        mask_mode=args.mask_mode,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )
    datamodule.setup("fit")
    assert datamodule.dataset is not None

    module = TerrainRepairLightningModule(
        num_material_classes=datamodule.dataset.num_material_classes,
        learning_rate=args.learning_rate,
        channels_last=args.channels_last,
    )
    if args.resume is not None:
        load_repair_checkpoint(args.resume, module.model, map_location="cpu")
        print(f"Loaded repair weights from {Path(args.resume).expanduser().resolve()}")
    if args.compile:
        module.model = torch.compile(module.model, mode=args.compile_mode)
        print(f"Using torch.compile(mode={args.compile_mode!r})")

    precision = args.precision or _precision_from_amp(args.amp)

    lightning_root = Path(args.lightning_root_dir).expanduser().resolve()
    lit_root = (
        Path(args.litlogger_root_dir).expanduser().resolve()
        if args.litlogger_root_dir
        else lightning_root / "litlogger"
    )
    metadata: dict[str, str] = {}
    for pair in args.litlogger_metadata or []:
        if "=" in pair:
            k, v = pair.split("=", 1)
            metadata[k.strip()] = v.strip()

    lit_logger = RepairLitLogger(
        root_dir=str(lit_root),
        name=args.litlogger_name,
        teamspace=args.litlogger_teamspace,
        metadata=metadata or None,
        log_model=args.litlogger_log_model,
        save_logs=not args.no_litlogger_save_logs,
    )

    lightning_ckpt_dir = lightning_root / "checkpoints"
    lightning_checkpoint = ModelCheckpoint(
        dirpath=lightning_ckpt_dir,
        filename="repair-{epoch:04d}-{step}",
        save_last=True,
        every_n_epochs=max(1, args.save_every),
    )

    validation_callback = RepairValidationCaseCallback(
        cases_dir=args.validation_cases_dir,
        validate_every=args.validate_every,
        channels_last=args.channels_last,
    )
    compatible_checkpoint = RepairCompatibleCheckpointCallback(
        checkpoint_path=args.checkpoint,
        latest_checkpoint_path=args.latest_checkpoint or checkpoint_sibling(args.checkpoint, "latest"),
        best_checkpoint_path=args.best_checkpoint or checkpoint_sibling(args.checkpoint, "best"),
        args=args,
        datamodule=datamodule,
        validation_callback=validation_callback,
        save_every=args.save_every,
    )

    limit_train_batches: float | int | None = None
    if args.limit_train_batches is not None:
        raw = args.limit_train_batches
        limit_train_batches = float(raw) if "." in raw else int(raw)

    print(
        f"Training on {len(datamodule.dataset.export_dirs)} "
        f"world{'s' if len(datamodule.dataset.export_dirs) != 1 else ''}"
    )
    print(
        f"Lightning trainer: accelerator={args.accelerator} devices={args.devices} "
        f"strategy={args.strategy} precision={precision} windows={len(datamodule.dataset)} "
        f"num_workers={num_workers}"
    )
    print(
        f"LitLogger: root={lit_root} name={args.litlogger_name!r} "
        f"teamspace={args.litlogger_teamspace!r} log_model={args.litlogger_log_model}"
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        precision=precision,
        max_epochs=args.epochs,
        logger=lit_logger,
        callbacks=[
            RepairEpochCallback(),
            validation_callback,
            compatible_checkpoint,
            lightning_checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
        accumulate_grad_batches=max(1, args.grad_accum_steps),
        gradient_clip_val=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        benchmark=args.cudnn_benchmark,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=str(lightning_root),
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=limit_train_batches,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()


__all__ = [
    "TerrainRepairDataModule",
    "TerrainRepairLightningModule",
    "main",
]