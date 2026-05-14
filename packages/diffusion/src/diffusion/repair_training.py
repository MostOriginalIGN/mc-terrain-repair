"""Training utilities for deterministic terrain repair."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .repair_data import TerrainRepairDataset
from .repair_model import TerrainRepairUNet


@dataclass
class RepairLossOutput:
    total_loss: torch.Tensor
    height_loss: torch.Tensor
    gradient_loss: torch.Tensor
    seam_loss: torch.Tensor
    material_loss: torch.Tensor
    support_loss: torch.Tensor


@dataclass(frozen=True)
class RepairTrainingState:
    completed_epochs: int = 0
    global_step: int = 0


@dataclass(frozen=True)
class RepairLossWeights:
    height: float = 1.0
    gradient: float = 0.5
    seam: float = 0.5
    material: float = 0.2
    support: float = 0.1


def charbonnier(error: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(error * error + eps * eps)


def height_gradients(height: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = F.pad(height[:, :, :, 1:] - height[:, :, :, :-1], (0, 1, 0, 0))
    grad_y = F.pad(height[:, :, 1:, :] - height[:, :, :-1, :], (0, 0, 0, 1))
    return grad_x, grad_y


def boundary_band(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def compute_repair_losses(
    model: TerrainRepairUNet,
    batch: dict[str, torch.Tensor],
    weights: RepairLossWeights = RepairLossWeights(),
) -> RepairLossOutput:
    target_height = batch["target_height"]
    target_material = batch["target_material"]
    target_support = batch["target_support"]
    known_height = batch["known_height"]
    known_material = batch["known_material"]
    known_support = batch["known_support"]
    mask = batch["mask"]
    prefill_height = batch["prefill_height"]
    boundary_distance = batch["boundary_distance"]
    prefill_gradients = batch["prefill_gradients"]
    prefill_laplacian = batch["prefill_laplacian"]

    outputs = model(
        known_height=known_height,
        prefill_height=prefill_height,
        mask=mask,
        known_material=known_material,
        known_support=known_support,
        boundary_distance=boundary_distance,
        prefill_gradients=prefill_gradients,
        prefill_laplacian=prefill_laplacian,
    )
    predicted_height = prefill_height + outputs.height_residual
    composite_height = target_height * (1.0 - mask) + predicted_height * mask

    height_loss = masked_mean(charbonnier(predicted_height - target_height), mask)

    pred_grad_x, pred_grad_y = height_gradients(composite_height)
    target_grad_x, target_grad_y = height_gradients(target_height)
    gradient_loss = masked_mean(
        charbonnier(pred_grad_x - target_grad_x) + charbonnier(pred_grad_y - target_grad_y),
        mask,
    )

    seam_mask = boundary_band(mask)
    seam_loss = masked_mean(
        charbonnier(pred_grad_x - target_grad_x) + charbonnier(pred_grad_y - target_grad_y),
        seam_mask,
    )

    material_loss_map = nn.functional.cross_entropy(outputs.material_logits, target_material, reduction="none")
    material_loss = masked_mean(material_loss_map.unsqueeze(1), mask)
    support_loss = masked_mean((outputs.support - target_support) ** 2, mask)

    total_loss = (
        weights.height * height_loss
        + weights.gradient * gradient_loss
        + weights.seam * seam_loss
        + weights.material * material_loss
        + weights.support * support_loss
    )
    return RepairLossOutput(
        total_loss=total_loss,
        height_loss=height_loss,
        gradient_loss=gradient_loss,
        seam_loss=seam_loss,
        material_loss=material_loss,
        support_loss=support_loss,
    )


def train_repair_step(
    model: TerrainRepairUNet,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    weights: RepairLossWeights = RepairLossWeights(),
) -> RepairLossOutput:
    optimizer.zero_grad(set_to_none=True)
    losses = compute_repair_losses(model, batch, weights=weights)
    losses.total_loss.backward()
    optimizer.step()
    return losses


def save_repair_checkpoint(
    path: str | Path,
    model: TerrainRepairUNet,
    optimizer: torch.optim.Optimizer | None,
    meta: dict[str, object] | None = None,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "num_material_classes": model.num_material_classes,
        "meta": meta or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_repair_checkpoint(
    path: str | Path,
    model: TerrainRepairUNet,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def restore_repair_training_state(payload: dict[str, object]) -> RepairTrainingState:
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return RepairTrainingState()
    return RepairTrainingState(
        completed_epochs=int(meta.get("epoch", 0) or 0),
        global_step=int(meta.get("global_step", 0) or 0),
    )


def build_repair_checkpoint_meta(
    args: argparse.Namespace,
    dataset: TerrainRepairDataset,
    state: RepairTrainingState,
    interrupted: bool,
) -> dict[str, object]:
    return {
        "model_type": "deterministic_repair_v1",
        "tile_size": args.tile_size,
        "stride_chunks": args.stride_chunks,
        "height_min": dataset.height_min,
        "height_max": dataset.height_max,
        "export_dir": str(Path(args.export_dir).resolve()),
        "epoch": state.completed_epochs,
        "global_step": state.global_step,
        "interrupted": interrupted,
    }


def persist_repair_checkpoint(
    checkpoint_path: str | Path,
    model: TerrainRepairUNet,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    dataset: TerrainRepairDataset,
    state: RepairTrainingState,
    interrupted: bool,
) -> None:
    save_repair_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        meta=build_repair_checkpoint_meta(args, dataset, state, interrupted=interrupted),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deterministic surface terrain repair.")
    parser.add_argument("--export-dir", required=True, help="Directory containing exported chunk and surface arrays")
    parser.add_argument("--checkpoint", required=True, help="Output checkpoint path")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=1, help="Total target epoch count, including resumed epochs")
    parser.add_argument("--save-every", type=int, default=1, help="Save a checkpoint every N epochs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--stride-chunks", type=int, default=1)
    parser.add_argument("--mask-mode", default="mixed", choices=["none", "rectangle", "strip", "blob", "mixed"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TerrainRepairDataset(
        args.export_dir,
        tile_size=args.tile_size,
        stride_chunks=args.stride_chunks,
        mask_mode=args.mask_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = TerrainRepairUNet(num_material_classes=dataset.num_material_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    weights = RepairLossWeights()

    state = RepairTrainingState()
    if args.resume is not None:
        payload = load_repair_checkpoint(args.resume, model, optimizer=optimizer, map_location=device)
        state = restore_repair_training_state(payload)
        print(
            f"Resumed repair model from {Path(args.resume).resolve()} at epoch "
            f"{state.completed_epochs} step {state.global_step}"
        )

    if state.completed_epochs >= args.epochs:
        print(
            f"Checkpoint already reached epoch {state.completed_epochs}, which meets or exceeds --epochs {args.epochs}. "
            "Nothing to do."
        )
        return

    model.train()
    progress: tqdm | None = None
    try:
        for epoch in range(state.completed_epochs, args.epochs):
            losses: RepairLossOutput | None = None
            progress = tqdm(loader, desc=f"Repair Epoch {epoch + 1}/{args.epochs}", unit="batch", dynamic_ncols=True)
            for batch in progress:
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                losses = train_repair_step(model, optimizer, batch, weights=weights)
                state = RepairTrainingState(completed_epochs=epoch, global_step=state.global_step + 1)
                progress.set_postfix({
                    "total": f"{losses.total_loss.item():.4f}",
                    "height": f"{losses.height_loss.item():.4f}",
                    "material": f"{losses.material_loss.item():.4f}",
                    "support": f"{losses.support_loss.item():.4f}",
                    "step": state.global_step,
                })
            progress.close()
            progress = None
            state = RepairTrainingState(completed_epochs=epoch + 1, global_step=state.global_step)
            if losses is not None:
                print(
                    f"repair epoch {epoch + 1}/{args.epochs}: total={losses.total_loss.item():.4f} "
                    f"height={losses.height_loss.item():.4f} gradient={losses.gradient_loss.item():.4f} "
                    f"seam={losses.seam_loss.item():.4f} material={losses.material_loss.item():.4f} "
                    f"support={losses.support_loss.item():.4f}"
                )
            if args.save_every > 0 and ((epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs):
                persist_repair_checkpoint(args.checkpoint, model, optimizer, args, dataset, state, interrupted=False)
                print(f"Saved repair checkpoint to {Path(args.checkpoint).resolve()}")
    except KeyboardInterrupt:
        if progress is not None:
            progress.close()
        persist_repair_checkpoint(args.checkpoint, model, optimizer, args, dataset, state, interrupted=True)
        print(f"Interrupted. Saved repair checkpoint to {Path(args.checkpoint).resolve()}")
        return

    if args.save_every <= 0:
        persist_repair_checkpoint(args.checkpoint, model, optimizer, args, dataset, state, interrupted=False)
        print(f"Saved repair checkpoint to {Path(args.checkpoint).resolve()}")


if __name__ == "__main__":
    main()


__all__ = [
    "RepairLossOutput",
    "RepairLossWeights",
    "RepairTrainingState",
    "build_repair_checkpoint_meta",
    "charbonnier",
    "compute_repair_losses",
    "load_repair_checkpoint",
    "restore_repair_training_state",
    "save_repair_checkpoint",
    "train_repair_step",
]
