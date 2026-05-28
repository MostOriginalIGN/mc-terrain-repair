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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, LitLogger
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from unet.repair_data import TerrainRepairDataset
from unet.repair_model import TerrainRepairUNet
from unet.repair_training import (
    RepairAmpConfig,
    RepairLossOutput,
    RepairLossWeights,
    RepairTrainingState,
    add_repair_loss_weight_args,
    build_repair_checkpoint_meta,
    checkpoint_sibling,
    configure_training_seed,
    compute_repair_losses,
    configure_cuda_backend,
    evaluate_repair_cases,
    load_repair_checkpoint,
    print_validation_overlap_warnings,
    repair_loss_weights_from_args,
    resolve_training_export_dirs,
    save_repair_checkpoint,
)

warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=FutureWarning,
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
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16-mixed"
    if torch.cuda.is_available():
        return "16-mixed"
    return "32-true"


def _loss_weights_hparams(weights: RepairLossWeights) -> dict[str, float]:
    return {
        "height": weights.height,
        "edge_height": weights.edge_height,
        "gradient": weights.gradient,
        "seam": weights.seam,
        "laplacian": weights.laplacian,
        "highpass": weights.highpass,
        "roughness": weights.roughness,
        "context": weights.context,
        "material": weights.material,
        "support": weights.support,
    }


def _log_loss_metrics(
    module: pl.LightningModule,
    prefix: str,
    losses: RepairLossOutput,
    *,
    batch_size: int,
    on_step: bool,
    on_epoch: bool,
) -> None:
    total_name = "score" if prefix == "val_window" else "total_loss"
    entries = (
        (total_name, "total_loss"),
        ("height_loss", "height_loss"),
        ("height_mae_blocks", "height_mae_blocks"),
        ("height_within_1_block", "height_within_1_block"),
        ("height_within_2_blocks", "height_within_2_blocks"),
        ("edge_height_loss", "edge_height_loss"),
        ("gradient_loss", "gradient_loss"),
        ("gradient_mae_blocks", "gradient_mae_blocks"),
        ("seam_loss", "seam_loss"),
        ("laplacian_loss", "laplacian_loss"),
        ("highpass_loss", "highpass_loss"),
        ("roughness_loss", "roughness_loss"),
        ("context_loss", "context_loss"),
        ("material_loss", "material_loss"),
        ("support_loss", "support_loss"),
    )
    for metric_name, attribute in entries:
        module.log(
            f"{prefix}/{metric_name}",
            getattr(losses, attribute),
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=metric_name == total_name or (prefix == "val_window" and metric_name == "height_mae_blocks"),
            batch_size=batch_size,
        )


def split_spatial_window_indices(
    dataset: TerrainRepairDataset,
    val_fraction: float,
    buffer_chunks: int,
) -> tuple[list[int], list[int]]:
    """Split windows by a held-out spatial slab with a buffer to avoid overlap leakage."""
    if val_fraction <= 0.0:
        return list(range(len(dataset))), []
    val_fraction = min(max(float(val_fraction), 0.0), 0.5)
    buffer_chunks = max(0, int(buffer_chunks))
    train_indices: list[int] = []
    val_indices: list[int] = []
    grouped: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for index, (export_id, origin) in enumerate(zip(dataset.window_export_ids, dataset.window_origins, strict=True)):
        grouped.setdefault(int(export_id), []).append((index, origin))

    for group in grouped.values():
        if len(group) < 2:
            train_indices.extend(index for index, _ in group)
            continue
        xs = [origin[0] for _, origin in group]
        zs = [origin[1] for _, origin in group]
        axis = 0 if max(xs) - min(xs) >= max(zs) - min(zs) else 1
        sorted_origins = sorted({origin[axis] for _, origin in group})
        split_position = max(1, min(len(sorted_origins) - 1, int(round(len(sorted_origins) * (1.0 - val_fraction)))))
        val_start = sorted_origins[split_position]
        train_limit = val_start - dataset.chunks_per_side - buffer_chunks + 1

        group_train = [index for index, origin in group if origin[axis] < train_limit]
        group_val = [index for index, origin in group if origin[axis] >= val_start]
        if not group_train or not group_val:
            train_indices.extend(index for index, _ in group)
            continue
        train_indices.extend(group_train)
        val_indices.extend(group_val)

    return sorted(train_indices), sorted(val_indices)


def _apply_window_subset(dataset: TerrainRepairDataset, indices: list[int]) -> None:
    dataset.window_origins = [dataset.window_origins[index] for index in indices]
    dataset.window_export_ids = [dataset.window_export_ids[index] for index in indices]


class TerrainRepairLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_material_classes: int,
        learning_rate: float = 1e-4,
        lr_scheduler: str = "none",
        weight_decay: float = 1e-2,
        channels_last: bool = False,
        model_base_channels: int = 64,
        model_depth: int = 4,
        model_bottleneck_dilations: str = "1,2,4,2",
        dropout: float = 0.0,
        weights: RepairLossWeights = RepairLossWeights(),
    ):
        super().__init__()
        self.save_hyperparameters({
            "num_material_classes": num_material_classes,
            "learning_rate": learning_rate,
            "lr_scheduler": lr_scheduler,
            "weight_decay": weight_decay,
            "channels_last": channels_last,
            "model_base_channels": model_base_channels,
            "model_depth": model_depth,
            "model_bottleneck_dilations": model_bottleneck_dilations,
            "dropout": dropout,
            "loss_weights": _loss_weights_hparams(weights),
        })
        self.model = TerrainRepairUNet(
            num_material_classes=num_material_classes,
            base_channels=model_base_channels,
            depth=model_depth,
            bottleneck_dilations=model_bottleneck_dilations,
            dropout=dropout,
        )
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.weight_decay = float(weight_decay)
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
        if getattr(self, "_trainer", None) is not None:
            _log_loss_metrics(self, "train", losses, batch_size=batch_size, on_step=True, on_epoch=True)
        return losses.total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch = _format_repair_batch(batch, self.channels_last)
        losses = compute_repair_losses(self.model, batch, weights=self.weights)
        batch_size = batch["target_height"].shape[0]
        _log_loss_metrics(self, "val_window", losses, batch_size=batch_size, on_step=False, on_epoch=True)
        return losses.total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.lr_scheduler == "none":
            return optimizer
        if self.lr_scheduler != "cosine":
            raise ValueError(f"Unsupported lr_scheduler: {self.lr_scheduler}")
        trainer = getattr(self, "_trainer", None)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=trainer.max_epochs if trainer is not None and trainer.max_epochs is not None else 100,
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
        augment: bool,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int | None = 4,
        prefill_iterations: int = 64,
        seed: int = 0,
        val_split: str = "spatial",
        val_fraction: float = 0.1,
        val_buffer_chunks: int = 8,
    ):
        super().__init__()
        self.export_dirs = export_dirs
        self.tile_size = tile_size
        self.stride_chunks = stride_chunks
        self.mask_mode = mask_mode
        self.augment = bool(augment)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.prefill_iterations = prefill_iterations
        self.seed = int(seed)
        self.val_split = val_split
        self.val_fraction = float(val_fraction)
        self.val_buffer_chunks = int(val_buffer_chunks)
        self.dataset: TerrainRepairDataset | None = None
        self.val_dataset: TerrainRepairDataset | None = None
        self.dropped_buffer_windows = 0

    def setup(self, stage: str | None = None) -> None:
        if self.dataset is None:
            full_dataset = TerrainRepairDataset(
                self.export_dirs,
                tile_size=self.tile_size,
                stride_chunks=self.stride_chunks,
                mask_mode=self.mask_mode,
                augment=self.augment,
                seed=self.seed,
                prefill_iterations=self.prefill_iterations,
            )
            if self.val_split == "spatial":
                all_count = len(full_dataset)
                train_indices, val_indices = split_spatial_window_indices(
                    full_dataset,
                    val_fraction=self.val_fraction,
                    buffer_chunks=self.val_buffer_chunks,
                )
                self.dropped_buffer_windows = all_count - len(train_indices) - len(val_indices)
                _apply_window_subset(full_dataset, train_indices)
                self.dataset = full_dataset
                if val_indices:
                    self.val_dataset = TerrainRepairDataset(
                        self.export_dirs,
                        tile_size=self.tile_size,
                        stride_chunks=self.stride_chunks,
                        mask_mode=self.mask_mode,
                        augment=False,
                        seed=self.seed + 10_000,
                        cache_arrays=False,
                        height_range=(full_dataset.height_min, full_dataset.height_max),
                        prefill_iterations=self.prefill_iterations,
                    )
                    _apply_window_subset(self.val_dataset, val_indices)
            else:
                self.dataset = full_dataset

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
            "generator": torch.Generator().manual_seed(self.seed),
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor or 2
        return DataLoader(self.dataset, **loader_kwargs)

    def val_dataloader(self) -> DataLoader | None:
        if self.dataset is None:
            self.setup("fit")
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None
        loader_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor or 2
        return DataLoader(self.val_dataset, **loader_kwargs)

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

        metrics = evaluate_repair_cases(
            pl_module.model,
            self.cases_dir,
            device=pl_module.device,
            amp_config=RepairAmpConfig(enabled=amp_dtype is not None, dtype=amp_dtype),
            channels_last=self.channels_last,
        )
        if metrics is None:
            return

        self.best_score = min(self.best_score, metrics.score)
        pl_module.log("val/score",             metrics.score,             sync_dist=False)
        pl_module.log("val/visual_score",      metrics.visual_score,      sync_dist=False)
        pl_module.log("val/legacy_score",      metrics.legacy_score,      sync_dist=False)
        pl_module.log("val/height_mae",        metrics.height_mae,        sync_dist=False)
        pl_module.log("val/height_mae_blocks", metrics.height_mae_blocks, sync_dist=False)
        pl_module.log("val/height_within_1_block", metrics.height_within_1_block, sync_dist=False)
        pl_module.log("val/height_within_2_blocks", metrics.height_within_2_blocks, sync_dist=False)
        pl_module.log("val/seam_mae",          metrics.seam_mae,          sync_dist=False)
        pl_module.log("val/seam_mae_blocks",   metrics.seam_mae_blocks,   sync_dist=False)
        pl_module.log("val/gradient_mae",      metrics.gradient_mae,      sync_dist=False)
        pl_module.log("val/gradient_mae_blocks", metrics.gradient_mae_blocks, sync_dist=False)
        pl_module.log("val/laplacian_mae",     metrics.laplacian_mae,     sync_dist=False)
        pl_module.log("val/laplacian_mae_blocks", metrics.laplacian_mae_blocks, sync_dist=False)
        pl_module.log("val/highpass_mae",      metrics.highpass_mae,      sync_dist=False)
        pl_module.log("val/highpass_mae_blocks", metrics.highpass_mae_blocks, sync_dist=False)
        pl_module.log("val/roughness_ratio",   metrics.roughness_ratio,   sync_dist=False)
        pl_module.log("val/context_style_error_blocks", metrics.context_style_error_blocks, sync_dist=False)
        pl_module.log("val/context_roughness_ratio", metrics.context_roughness_ratio, sync_dist=False)
        pl_module.log("val/context_laplacian_ratio", metrics.context_laplacian_ratio, sync_dist=False)
        pl_module.log("val/context_highpass_ratio", metrics.context_highpass_ratio, sync_dist=False)
        pl_module.log("val/material_accuracy", metrics.material_accuracy,  sync_dist=False)
        pl_module.log("val/support_mse",       metrics.support_mse,       sync_dist=False)
        print(
            f"repair validation: visual_score={metrics.visual_score:.4f} legacy_score={metrics.legacy_score:.4f} "
            f"height_blocks={metrics.height_mae_blocks:.3f} within1={metrics.height_within_1_block:.3f} "
            f"within2={metrics.height_within_2_blocks:.3f} seam_blocks={metrics.seam_mae_blocks:.3f} "
            f"grad_blocks={metrics.gradient_mae_blocks:.3f} lap_blocks={metrics.laplacian_mae_blocks:.3f} "
            f"highpass_blocks={metrics.highpass_mae_blocks:.3f} roughness_ratio={metrics.roughness_ratio:.3f} "
            f"context_blocks={metrics.context_style_error_blocks:.3f} context_rough={metrics.context_roughness_ratio:.3f} "
            f"material_acc={metrics.material_accuracy:.4f} support_mse={metrics.support_mse:.4f} "
            f"cases={metrics.case_count} mask_pixels={metrics.mask_pixel_count}"
        )


class RepairCompatibleCheckpointCallback(pl.Callback):
    """Save repair.pt-compatible checkpoints alongside the Lightning .ckpt files."""

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

    def _current_score(self, trainer: pl.Trainer) -> float:
        if self.validation_callback.best_score < float("inf"):
            return self.validation_callback.best_score
        val_window_score = trainer.callback_metrics.get("val_window/score")
        if isinstance(val_window_score, torch.Tensor) and torch.isfinite(val_window_score):
            return float(val_window_score.detach().cpu())
        return float("inf")

    def _save(
        self,
        path: Path,
        trainer: pl.Trainer,
        pl_module: TerrainRepairLightningModule,
        interrupted: bool,
        completed_epochs: int | None = None,
    ) -> None:
        dataset = self.datamodule.dataset
        if dataset is None:
            return
        state = RepairTrainingState(
            completed_epochs=trainer.current_epoch if completed_epochs is None else completed_epochs,
            global_step=trainer.global_step,
        )
        score = self._current_score(trainer)
        export_args = argparse.Namespace(**vars(self.args), best_score=score if score < float("inf") else None)
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

        self._save(self.checkpoint_path, trainer, pl_module, interrupted=False, completed_epochs=completed_epoch)
        if self.latest_checkpoint_path != self.checkpoint_path:
            self._save(self.latest_checkpoint_path, trainer, pl_module, interrupted=False, completed_epochs=completed_epoch)

        candidate_score = self._current_score(trainer)
        if candidate_score < self.best_score:
            self.best_score = candidate_score
            self._save(self.best_checkpoint_path, trainer, pl_module, interrupted=False, completed_epochs=completed_epoch)

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
    parser.add_argument("--seed", type=int, default=0, help="Seed for masks, shuffling, workers, and model initialization.")
    parser.add_argument("--lr-scheduler", default="none", choices=["none", "cosine"],
                        help="Learning-rate schedule. Default preserves the legacy constant LR behavior.")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--model-base-channels", type=int, default=64)
    parser.add_argument("--model-depth", type=int, default=4)
    parser.add_argument(
        "--model-bottleneck-dilations",
        default="1,2,4,2",
        help="Comma-separated dilation rates for bottleneck residual blocks. Empty disables extra dilated blocks.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--stride-chunks", type=int, default=1)
    parser.add_argument("--mask-mode", default="selection_mixed",
                        choices=["none", "rectangle", "strip", "blob", "mixed", "terrain_mixed", "selection_mixed"])
    parser.add_argument("--augment", action="store_true", help="Enable random flip/rot90 spatial augmentation.")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)

    parser.add_argument("--amp", default="auto", choices=["auto", "off", "fp16", "bf16"])
    parser.add_argument("--precision", default=None,
                        help="Lightning precision string (overrides --amp): 16-mixed, bf16-mixed, 32-true, …")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=-1,
                        help="DataLoader workers: -1 → min(8, cpu_count-1), 0 → main process only.")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="Batches prefetched per DataLoader worker. Ignored when --num-workers=0.")
    parser.add_argument("--prefill-iterations", type=int, default=64,
                        help="Neighbor-averaging iterations used to build masked height prefill.")
    parser.add_argument("--matmul-precision", default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--tf32", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])

    parser.add_argument("--litlogger-root-dir", default=None,
                        help="Root directory for LitLogger local files (default: <lightning-root-dir>/litlogger).")
    parser.add_argument("--logger", default="csv", choices=["csv", "litlogger", "none"],
                        help="Training logger. Default csv is local/offline; litlogger requires Lightning credentials.")
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
    parser.add_argument("--val-split", default="spatial", choices=["none", "spatial"],
                        help="Automatic train/validation split over export windows. Spatial uses a buffered held-out slab.")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of discovered spatial window origins to hold out per export for --val-split=spatial.")
    parser.add_argument("--val-buffer-chunks", type=int, default=8,
                        help="Chunk gap between train and validation origin slabs for spatial split.")
    add_repair_loss_weight_args(parser)

    parser.add_argument("--lightning-root-dir", default="./artifacts/lightning")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--limit-train-batches", default=None,
                        help="Fraction (0–1) or absolute count of batches per epoch.")
    parser.add_argument("--limit-val-batches", default=None,
                        help="Fraction (0–1) or absolute count of validation batches per epoch.")
    parser.add_argument("--fast-dev-run", action="store_true")

    args = parser.parse_args()

    configure_training_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)
    if args.early_stopping_patience > 0 and args.validation_cases_dir is None and args.val_split == "none":
        raise SystemExit("--early-stopping-patience requires --validation-cases-dir or --val-split so a score can be monitored.")

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
        augment=args.augment,
        batch_size=args.batch_size,
        num_workers=num_workers,
        prefetch_factor=args.prefetch_factor,
        prefill_iterations=args.prefill_iterations,
        seed=args.seed,
        val_split=args.val_split,
        val_fraction=args.val_fraction,
        val_buffer_chunks=args.val_buffer_chunks,
    )
    datamodule.setup("fit")
    assert datamodule.dataset is not None
    print_validation_overlap_warnings(args.validation_cases_dir, export_dirs)

    module = TerrainRepairLightningModule(
        num_material_classes=datamodule.dataset.num_material_classes,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        weight_decay=args.weight_decay,
        channels_last=args.channels_last,
        model_base_channels=args.model_base_channels,
        model_depth=args.model_depth,
        model_bottleneck_dilations=args.model_bottleneck_dilations,
        dropout=args.dropout,
        weights=repair_loss_weights_from_args(args),
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

    logger: CSVLogger | RepairLitLogger | bool
    if args.logger == "litlogger":
        logger = RepairLitLogger(
            root_dir=str(lit_root),
            name=args.litlogger_name,
            teamspace=args.litlogger_teamspace,
            metadata=metadata or None,
            log_model=args.litlogger_log_model,
            save_logs=not args.no_litlogger_save_logs,
        )
    elif args.logger == "csv":
        logger = CSVLogger(save_dir=str(lightning_root), name=args.litlogger_name or "repair")
    else:
        logger = False

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
    limit_val_batches: float | int | None = None
    if args.limit_val_batches is not None:
        raw = args.limit_val_batches
        limit_val_batches = float(raw) if "." in raw else int(raw)

    print(
        f"Training on {len(datamodule.dataset.export_dirs)} "
        f"world{'s' if len(datamodule.dataset.export_dirs) != 1 else ''}"
    )
    if datamodule.val_dataset is not None:
        print(
            f"Spatial validation split: train_windows={len(datamodule.dataset)} "
            f"val_windows={len(datamodule.val_dataset)} dropped_buffer_windows={datamodule.dropped_buffer_windows}"
        )
    elif args.val_split != "none":
        print("Spatial validation split: disabled because no non-overlapping validation windows were available.")
    print(
        f"Lightning trainer: accelerator={args.accelerator} devices={args.devices} "
        f"strategy={args.strategy} precision={precision} windows={len(datamodule.dataset)} "
        f"num_workers={num_workers}"
    )
    print(
        f"Logger: {args.logger} root={lit_root if args.logger == 'litlogger' else lightning_root} "
        f"name={args.litlogger_name!r} teamspace={args.litlogger_teamspace!r} "
        f"log_model={args.litlogger_log_model}"
    )

    callbacks: list[pl.Callback] = [
        RepairEpochCallback(),
        validation_callback,
    ]
    if args.early_stopping_patience > 0:
        early_stopping_monitor = "val/score" if args.validation_cases_dir is not None else "val_window/score"
        callbacks.append(EarlyStopping(
            monitor=early_stopping_monitor,
            mode="min",
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
        ))
    callbacks.extend([
        compatible_checkpoint,
        lightning_checkpoint,
    ])
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        precision=precision,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=max(1, args.grad_accum_steps),
        gradient_clip_val=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        benchmark=args.cudnn_benchmark,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=str(lightning_root),
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()


__all__ = [
    "TerrainRepairDataModule",
    "TerrainRepairLightningModule",
    "main",
    "split_spatial_window_indices",
]
