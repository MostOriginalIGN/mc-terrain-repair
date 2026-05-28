"""Training utilities for deterministic terrain repair."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from exporter.vocab import UNKNOWN_INDEX

from .repair_data import (
    TerrainRepairDataset,
    build_prefill_height,
    compute_boundary_distance,
    compute_height_gradients,
    compute_laplacian,
    estimate_support_from_material,
)
from .repair_model import TerrainRepairUNet, TerrainRepairUNetV1


def is_export_dir(path: str | Path) -> bool:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        return False
    return any(candidate.glob("surface_*.npy")) and any(candidate.glob("chunk_*.npy"))


def resolve_training_export_dirs(export_dirs: list[str]) -> list[Path]:
    resolved_inputs = TerrainRepairDataset._resolve_export_dirs(export_dirs)
    expanded: list[Path] = []
    for path in resolved_inputs:
        if is_export_dir(path):
            if path not in expanded:
                expanded.append(path)
            continue
        child_exports = sorted(
            child.resolve()
            for child in path.iterdir()
            if child.is_dir() and is_export_dir(child)
        )
        if child_exports:
            for child in child_exports:
                if child not in expanded:
                    expanded.append(child)
            continue
        raise SystemExit(
            f"No export directories found at {path}. "
            "Pass an export directory containing surface_*.npy/chunk_*.npy files, "
            "or a parent directory whose immediate children are export directories."
        )
    return expanded


@dataclass
class RepairLossOutput:
    total_loss: torch.Tensor
    height_loss: torch.Tensor
    edge_height_loss: torch.Tensor
    gradient_loss: torch.Tensor
    seam_loss: torch.Tensor
    laplacian_loss: torch.Tensor
    highpass_loss: torch.Tensor
    roughness_loss: torch.Tensor
    context_loss: torch.Tensor
    material_loss: torch.Tensor
    support_loss: torch.Tensor
    height_mae_blocks: torch.Tensor
    height_within_1_block: torch.Tensor
    height_within_2_blocks: torch.Tensor
    gradient_mae_blocks: torch.Tensor


@dataclass(frozen=True)
class RepairTrainingState:
    completed_epochs: int = 0
    global_step: int = 0


@dataclass(frozen=True)
class RepairLossWeights:
    height: float = 1.0
    edge_height: float = 0.35
    gradient: float = 0.75
    seam: float = 0.75
    laplacian: float = 0.2
    highpass: float = 0.15
    roughness: float = 0.15
    context: float = 0.25
    material: float = 0.05
    support: float = 0.1


@dataclass(frozen=True)
class RepairAmpConfig:
    enabled: bool
    dtype: torch.dtype | None


@dataclass(frozen=True)
class RepairValidationMetrics:
    score: float
    visual_score: float
    legacy_score: float
    height_mae: float
    height_mae_blocks: float
    height_within_1_block: float
    height_within_2_blocks: float
    seam_mae: float
    seam_mae_blocks: float
    gradient_mae: float
    gradient_mae_blocks: float
    laplacian_mae: float
    laplacian_mae_blocks: float
    highpass_mae: float
    highpass_mae_blocks: float
    roughness_ratio: float
    context_style_error_blocks: float
    context_roughness_ratio: float
    context_laplacian_ratio: float
    context_highpass_ratio: float
    material_accuracy: float
    support_mse: float
    case_count: int
    mask_pixel_count: int


@dataclass(frozen=True)
class _ContextStyleMetrics:
    style_error_blocks: torch.Tensor
    roughness_ratio: torch.Tensor
    laplacian_ratio: torch.Tensor
    highpass_ratio: torch.Tensor


def charbonnier(error: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(error * error + eps * eps)


def height_gradients(height: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = F.pad(height[:, :, :, 1:] - height[:, :, :, :-1], (0, 1, 0, 0))
    grad_y = F.pad(height[:, :, 1:, :] - height[:, :, :-1, :], (0, 0, 0, 1))
    return grad_x, grad_y


def height_laplacian(height: torch.Tensor) -> torch.Tensor:
    padded = F.pad(height, (1, 1, 1, 1), mode="replicate")
    return (
        padded[:, :, 1:-1, :-2]
        + padded[:, :, 1:-1, 2:]
        + padded[:, :, :-2, 1:-1]
        + padded[:, :, 2:, 1:-1]
        - 4.0 * padded[:, :, 1:-1, 1:-1]
    )


def height_highpass(height: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be an odd value >= 3, got {kernel_size}")
    lowpass = F.avg_pool2d(
        height,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )
    return height - lowpass


def terrain_detail_features(height: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_x, grad_y = height_gradients(height)
    return (
        grad_x.abs() + grad_y.abs(),
        height_laplacian(height).abs(),
        height_highpass(height).abs(),
    )


def boundary_band(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def context_band(mask: torch.Tensor, radius: int = 8) -> torch.Tensor:
    kernel_size = max(3, int(radius) * 2 + 1)
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (dilated - mask).clamp(0.0, 1.0)


def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_sum(values, mask) / mask.sum().clamp(min=1.0)


def masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(values, mask)
    return torch.sqrt(masked_mean((values - mean) ** 2, mask) + 1e-8)


def height_scale_for_batch(batch: dict[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    raw = batch.get("height_scale")
    if raw is None:
        return torch.ones((reference.shape[0], 1, 1, 1), device=reference.device, dtype=reference.dtype)
    scale = raw.to(device=reference.device, dtype=reference.dtype)
    if scale.ndim == 0:
        scale = scale.view(1, 1, 1, 1).expand(reference.shape[0], 1, 1, 1)
    elif scale.ndim == 1:
        scale = scale.view(-1, 1, 1, 1)
    elif scale.ndim == 3:
        scale = scale[:, None, :, :]
    return scale.clamp(min=1e-6)


def context_style_loss(
    composite_height: torch.Tensor,
    target_height: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    context_mask = context_band(mask)
    has_context = (context_mask.sum() > 0).to(dtype=composite_height.dtype)

    pred_features = terrain_detail_features(composite_height)
    context_features = terrain_detail_features(target_height)
    loss = composite_height.new_zeros(())
    for pred_feature, context_feature in zip(pred_features, context_features, strict=True):
        pred_mean = masked_mean(pred_feature, mask)
        pred_std = masked_std(pred_feature, mask)
        context_mean = masked_mean(context_feature, context_mask)
        context_std = masked_std(context_feature, context_mask)
        loss = loss + charbonnier(pred_mean - context_mean) + 0.5 * charbonnier(pred_std - context_std)
    return loss * has_context / len(pred_features)


def context_style_metrics(
    predicted_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    reference_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mask: torch.Tensor,
    height_scale: torch.Tensor,
) -> _ContextStyleMetrics:
    context_mask = context_band(mask)
    has_context = (context_mask.sum() > 0).to(dtype=height_scale.dtype, device=height_scale.device)
    one = height_scale.new_ones(())

    mean_deltas: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    for predicted_feature, reference_feature in zip(predicted_features, reference_features, strict=True):
        predicted_mean = masked_mean(predicted_feature, mask)
        context_mean = masked_mean(reference_feature, context_mask)
        mean_deltas.append((predicted_mean - context_mean).abs())
        ratio = predicted_mean / context_mean.clamp(min=1e-8)
        ratios.append(ratio * has_context + one * (1.0 - has_context))

    style_error = (mean_deltas[0] + 0.5 * mean_deltas[1] + 0.5 * mean_deltas[2]) * height_scale.mean()
    return _ContextStyleMetrics(
        style_error_blocks=style_error * has_context,
        roughness_ratio=ratios[0] if ratios else one,
        laplacian_ratio=ratios[1] if len(ratios) > 1 else one,
        highpass_ratio=ratios[2] if len(ratios) > 2 else one,
    )


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
    height_scale = height_scale_for_batch(batch, target_height)

    height_loss = masked_mean(charbonnier(predicted_height - target_height), mask)
    block_error = (predicted_height - target_height).abs() * height_scale
    height_mae_blocks = masked_mean(block_error, mask)
    height_within_1_block = masked_mean((block_error <= 1.0).float(), mask)
    height_within_2_blocks = masked_mean((block_error <= 2.0).float(), mask)

    pred_grad_x, pred_grad_y = height_gradients(composite_height)
    target_grad_x, target_grad_y = height_gradients(target_height)
    seam_mask = boundary_band(mask)
    inner_edge_mask = (mask * seam_mask).clamp(0.0, 1.0)
    edge_error = predicted_height - target_height
    edge_height_loss = masked_mean(charbonnier(edge_error), inner_edge_mask) + charbonnier(
        masked_mean(edge_error, inner_edge_mask)
    )
    gradient_loss = masked_mean(
        charbonnier(pred_grad_x - target_grad_x) + charbonnier(pred_grad_y - target_grad_y),
        mask,
    )
    gradient_mae_blocks = masked_mean(
        ((pred_grad_x - target_grad_x).abs() + (pred_grad_y - target_grad_y).abs()) * height_scale,
        mask,
    )

    seam_loss = masked_mean(
        charbonnier(pred_grad_x - target_grad_x) + charbonnier(pred_grad_y - target_grad_y),
        seam_mask,
    )
    laplacian_loss = masked_mean(
        charbonnier(height_laplacian(composite_height) - height_laplacian(target_height)),
        mask,
    )
    highpass_loss = masked_mean(
        charbonnier(height_highpass(composite_height) - height_highpass(target_height)),
        mask,
    )
    pred_roughness = pred_grad_x.abs() + pred_grad_y.abs()
    target_roughness = target_grad_x.abs() + target_grad_y.abs()
    roughness_loss = masked_mean(charbonnier(pred_roughness - target_roughness), mask)
    context_loss = context_style_loss(composite_height, target_height, mask)

    material_loss_map = nn.functional.cross_entropy(outputs.material_logits, target_material, reduction="none")
    material_loss = masked_mean(material_loss_map.unsqueeze(1), mask)
    support_loss = masked_mean((outputs.support - target_support) ** 2, mask)

    total_loss = (
        weights.height * height_loss
        + weights.edge_height * edge_height_loss
        + weights.gradient * gradient_loss
        + weights.seam * seam_loss
        + weights.laplacian * laplacian_loss
        + weights.highpass * highpass_loss
        + weights.roughness * roughness_loss
        + weights.context * context_loss
        + weights.material * material_loss
        + weights.support * support_loss
    )
    return RepairLossOutput(
        total_loss=total_loss,
        height_loss=height_loss,
        edge_height_loss=edge_height_loss,
        gradient_loss=gradient_loss,
        seam_loss=seam_loss,
        laplacian_loss=laplacian_loss,
        highpass_loss=highpass_loss,
        roughness_loss=roughness_loss,
        context_loss=context_loss,
        material_loss=material_loss,
        support_loss=support_loss,
        height_mae_blocks=height_mae_blocks,
        height_within_1_block=height_within_1_block,
        height_within_2_blocks=height_within_2_blocks,
        gradient_mae_blocks=gradient_mae_blocks,
    )


def train_repair_step(
    model: TerrainRepairUNet,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    weights: RepairLossWeights = RepairLossWeights(),
    amp_config: RepairAmpConfig = RepairAmpConfig(enabled=False, dtype=None),
    scaler: torch.amp.GradScaler | None = None,
    grad_clip_norm: float | None = None,
    loss_scale: float = 1.0,
    step_optimizer: bool = True,
    zero_grad: bool = True,
) -> RepairLossOutput:
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    device_type = next(model.parameters()).device.type
    with torch.autocast(device_type=device_type, dtype=amp_config.dtype, enabled=amp_config.enabled):
        losses = compute_repair_losses(model, batch, weights=weights)
        backward_loss = losses.total_loss / max(loss_scale, 1.0)
    if scaler is not None and scaler.is_enabled():
        scaler.scale(backward_loss).backward()
        if step_optimizer:
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
    else:
        backward_loss.backward()
        if step_optimizer:
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
    return losses


def resolve_amp_config(device: torch.device, requested: str) -> RepairAmpConfig:
    if requested == "off":
        return RepairAmpConfig(enabled=False, dtype=None)
    if requested == "fp16":
        return RepairAmpConfig(enabled=device.type == "cuda", dtype=torch.float16)
    if requested == "bf16":
        return RepairAmpConfig(enabled=device.type in {"cuda", "cpu"}, dtype=torch.bfloat16)
    if requested != "auto":
        raise ValueError(f"Unsupported AMP mode: {requested}")
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return RepairAmpConfig(enabled=True, dtype=torch.bfloat16)
        return RepairAmpConfig(enabled=True, dtype=torch.float16)
    return RepairAmpConfig(enabled=False, dtype=None)


def move_repair_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    channels_last: bool = False,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
            continue
        value = value.to(device, non_blocking=True)
        if channels_last and value.ndim == 4 and value.is_floating_point():
            value = value.contiguous(memory_format=torch.channels_last)
        moved[key] = value
    return moved


def select_training_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        if device.type == "mps" and (getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available()):
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_cuda_backend(device: torch.device, tf32: str, cudnn_benchmark: bool) -> None:
    if device.type != "cuda":
        return
    allow_tf32 = tf32 == "on" or tf32 == "auto"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark


def configure_training_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_summary_writer(log_dir: str | None):
    if log_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard is not installed; continuing without TensorBoard logging.")
        return None
    path = Path(log_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(path))


def add_repair_loss_weight_args(parser: argparse.ArgumentParser) -> None:
    defaults = RepairLossWeights()
    parser.add_argument("--loss-height", type=float, default=defaults.height)
    parser.add_argument("--loss-edge-height", type=float, default=defaults.edge_height)
    parser.add_argument("--loss-gradient", type=float, default=defaults.gradient)
    parser.add_argument("--loss-seam", type=float, default=defaults.seam)
    parser.add_argument("--loss-laplacian", type=float, default=defaults.laplacian)
    parser.add_argument("--loss-highpass", type=float, default=defaults.highpass)
    parser.add_argument("--loss-roughness", type=float, default=defaults.roughness)
    parser.add_argument("--loss-context", type=float, default=defaults.context)
    parser.add_argument("--loss-material", type=float, default=defaults.material)
    parser.add_argument("--loss-support", type=float, default=defaults.support)


def repair_loss_weights_from_args(args: argparse.Namespace) -> RepairLossWeights:
    defaults = RepairLossWeights()
    return RepairLossWeights(
        height=float(getattr(args, "loss_height", defaults.height)),
        edge_height=float(getattr(args, "loss_edge_height", defaults.edge_height)),
        gradient=float(getattr(args, "loss_gradient", defaults.gradient)),
        seam=float(getattr(args, "loss_seam", defaults.seam)),
        laplacian=float(getattr(args, "loss_laplacian", defaults.laplacian)),
        highpass=float(getattr(args, "loss_highpass", defaults.highpass)),
        roughness=float(getattr(args, "loss_roughness", defaults.roughness)),
        context=float(getattr(args, "loss_context", defaults.context)),
        material=float(getattr(args, "loss_material", defaults.material)),
        support=float(getattr(args, "loss_support", defaults.support)),
    )


def repair_loss_weights_payload(args: argparse.Namespace) -> dict[str, float]:
    weights = repair_loss_weights_from_args(args)
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


def save_repair_checkpoint(
    path: str | Path,
    model: TerrainRepairUNet | TerrainRepairUNetV1,
    optimizer: torch.optim.Optimizer | None,
    meta: dict[str, object] | None = None,
) -> None:
    checkpoint_meta = dict(meta or {})
    if hasattr(model, "checkpoint_config"):
        checkpoint_meta.update(model.checkpoint_config())
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "num_material_classes": model.num_material_classes,
        "meta": checkpoint_meta,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def _load_repair_checkpoint_payload(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location=map_location)
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
        raise RuntimeError(
            f"Could not load repair checkpoint at {path.expanduser().resolve()}. "
            "The file is missing, incomplete, or not a valid PyTorch repair checkpoint. "
            "If training is still running, wait for it to finish saving; otherwise rerun "
            "`make train` to create a fresh checkpoint."
        ) from exc
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError(
            f"Could not load repair checkpoint at {path.expanduser().resolve()}. "
            "This file does not look like a deterministic repair checkpoint. "
            "Use `make train` to create artifacts/repair.pt before running `make repair`."
        )
    return checkpoint


def load_repair_checkpoint(
    path: str | Path,
    model: TerrainRepairUNet | TerrainRepairUNetV1,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    checkpoint = _load_repair_checkpoint_payload(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def _model_kwargs_from_checkpoint(checkpoint: dict[str, object]) -> dict[str, object]:
    meta = checkpoint.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    return {
        "base_channels": int(meta.get("model_base_channels", 64) or 64),
        "depth": int(meta.get("model_depth", 4) or 4),
        "bottleneck_dilations": str(meta.get("model_bottleneck_dilations", "1,2,4,2") or ""),
        "dropout": float(meta.get("model_dropout", 0.0) or 0.0),
    }


def load_repair_model_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[TerrainRepairUNet | TerrainRepairUNetV1, dict[str, object]]:
    """Load a repair model, selecting v1 or v2 architecture from the checkpoint."""
    checkpoint = _load_repair_checkpoint_payload(path, map_location=map_location)
    state_dict = checkpoint["model_state"]
    num_material_classes = int(checkpoint.get("num_material_classes", 17))
    if isinstance(state_dict, dict) and any(str(key).startswith("down1.") for key in state_dict):
        model: TerrainRepairUNet | TerrainRepairUNetV1 = TerrainRepairUNetV1(
            num_material_classes=num_material_classes,
        )
    else:
        model = TerrainRepairUNet(
            num_material_classes=num_material_classes,
            **_model_kwargs_from_checkpoint(checkpoint),
        )
    model.load_state_dict(state_dict)
    return model, checkpoint


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
    export_dirs = [str(path) for path in dataset.export_dirs]
    model_depth = int(getattr(args, "model_depth", 4) or 4)
    model_base_channels = int(getattr(args, "model_base_channels", 64) or 64)
    model_bottleneck_dilations = str(getattr(args, "model_bottleneck_dilations", "1,2,4,2") or "")
    return {
        "model_type": "deterministic_repair_v2",
        "model_base_channels": model_base_channels,
        "model_depth": model_depth,
        "model_bottleneck_dilations": model_bottleneck_dilations,
        "tile_size": args.tile_size,
        "stride_chunks": args.stride_chunks,
        "height_min": dataset.height_min,
        "height_max": dataset.height_max,
        "export_dir": export_dirs[0],
        "export_dirs": export_dirs,
        "epoch": state.completed_epochs,
        "global_step": state.global_step,
        "interrupted": interrupted,
        "mask_mode": getattr(args, "mask_mode", None),
        "dropout": getattr(args, "dropout", 0.0),
        "weight_decay": getattr(args, "weight_decay", 1e-2),
        "augment": getattr(args, "augment", False),
        "lr_scheduler": getattr(args, "lr_scheduler", "none"),
        "learning_rate": getattr(args, "learning_rate", 1e-4),
        "seed": getattr(args, "seed", None),
        "amp": getattr(args, "amp", None),
        "channels_last": getattr(args, "channels_last", None),
        "compile": getattr(args, "compile", None),
        "best_score": getattr(args, "best_score", None),
        "validation_score_type": "visual_score",
        "loss_weights": repair_loss_weights_payload(args),
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


def persist_repair_checkpoints(
    checkpoint_path: str | Path,
    model: TerrainRepairUNet,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    dataset: TerrainRepairDataset,
    state: RepairTrainingState,
    interrupted: bool,
    latest_checkpoint_path: str | Path | None = None,
) -> None:
    persist_repair_checkpoint(checkpoint_path, model, optimizer, args, dataset, state, interrupted=interrupted)
    if latest_checkpoint_path is not None and Path(latest_checkpoint_path) != Path(checkpoint_path):
        persist_repair_checkpoint(latest_checkpoint_path, model, optimizer, args, dataset, state, interrupted=interrupted)


def checkpoint_sibling(path: str | Path, suffix: str) -> Path:
    checkpoint_path = Path(path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_{suffix}{checkpoint_path.suffix}")


def validation_overlap_warnings(
    validation_cases_dir: str | Path | None,
    training_export_dirs: list[Path],
) -> list[str]:
    if validation_cases_dir is None:
        return []
    cases_root = Path(validation_cases_dir).expanduser().resolve()
    if not cases_root.is_dir():
        return []
    training_exports = {path.expanduser().resolve() for path in training_export_dirs}
    overlapping_cases: list[str] = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        metadata_path = case_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        export_dir = metadata.get("export_dir")
        if not isinstance(export_dir, str):
            continue
        if Path(export_dir).expanduser().resolve() in training_exports:
            overlapping_cases.append(case_dir.name)
    if not overlapping_cases:
        return []
    names = ", ".join(overlapping_cases[:8])
    suffix = "" if len(overlapping_cases) <= 8 else f", +{len(overlapping_cases) - 8} more"
    return [
        "validation cases share export_dir metadata with training data "
        f"({names}{suffix}); treat val/score as qualitative unless these cases are held out spatially."
    ]


def print_validation_overlap_warnings(
    validation_cases_dir: str | Path | None,
    training_export_dirs: list[Path],
) -> None:
    for warning in validation_overlap_warnings(validation_cases_dir, training_export_dirs):
        print(f"Validation warning: {warning}")


def _load_case_tensor(path: Path, dtype: np.dtype | type[np.generic]) -> np.ndarray:
    return np.load(path).astype(dtype)


def _case_height_scale(case_dir: Path) -> float:
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.is_file():
        return 1.0
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1.0
    height_min = metadata.get("height_min")
    height_max = metadata.get("height_max")
    if isinstance(height_min, (int, float)) and isinstance(height_max, (int, float)) and height_max > height_min:
        return float(height_max) - float(height_min)
    return 1.0


def _build_validation_batch(case_dir: Path, device: torch.device, channels_last: bool) -> dict[str, torch.Tensor] | None:
    required = [
        case_dir / "known_height.npy",
        case_dir / "known_material.npy",
        case_dir / "mask.npy",
        case_dir / "target_height.npy",
        case_dir / "target_material.npy",
    ]
    if not all(path.is_file() for path in required):
        return None

    known_height = _load_case_tensor(case_dir / "known_height.npy", np.float32)
    known_material = _load_case_tensor(case_dir / "known_material.npy", np.int64)
    mask = _load_case_tensor(case_dir / "mask.npy", np.float32)
    target_height = _load_case_tensor(case_dir / "target_height.npy", np.float32)
    target_material = _load_case_tensor(case_dir / "target_material.npy", np.int64)
    target_support_path = case_dir / "target_support.npy"
    known_support_path = case_dir / "known_support.npy"
    target_support = (
        _load_case_tensor(target_support_path, np.float32)
        if target_support_path.is_file()
        else estimate_support_from_material(target_material)
    )
    known_support = (
        _load_case_tensor(known_support_path, np.float32)
        if known_support_path.is_file()
        else target_support * (1.0 - mask)
    )

    known_material = known_material.copy()
    known_material[mask.astype(bool)] = UNKNOWN_INDEX
    prefill_height = build_prefill_height(known_height, mask)
    height_scale = _case_height_scale(case_dir)
    batch = {
        "known_height": torch.from_numpy(known_height[None, None, ...]),
        "known_material": torch.from_numpy(known_material[None, ...]),
        "known_support": torch.from_numpy(known_support[None, None, ...]),
        "mask": torch.from_numpy(mask[None, None, ...]),
        "prefill_height": torch.from_numpy(prefill_height[None, None, ...]),
        "boundary_distance": torch.from_numpy(compute_boundary_distance(mask)[None, None, ...]),
        "prefill_gradients": torch.from_numpy(compute_height_gradients(prefill_height)[None, ...]),
        "prefill_laplacian": torch.from_numpy(compute_laplacian(prefill_height)[None, None, ...]),
        "target_height": torch.from_numpy(target_height[None, None, ...]),
        "target_material": torch.from_numpy(target_material[None, ...]),
        "target_support": torch.from_numpy(target_support[None, None, ...]),
        "height_scale": torch.tensor([height_scale], dtype=torch.float32),
    }
    return move_repair_batch(batch, device, channels_last=channels_last)


def evaluate_repair_cases(
    model: torch.nn.Module,
    cases_dir: str | Path,
    device: torch.device,
    amp_config: RepairAmpConfig = RepairAmpConfig(enabled=False, dtype=None),
    channels_last: bool = False,
) -> RepairValidationMetrics | None:
    root = Path(cases_dir).expanduser().resolve()
    if not root.is_dir():
        return None

    totals = {
        "height_mae": 0.0,
        "height_mae_blocks": 0.0,
        "height_within_1_block": 0.0,
        "height_within_2_blocks": 0.0,
        "seam_mae": 0.0,
        "seam_mae_blocks": 0.0,
        "gradient_mae": 0.0,
        "gradient_mae_blocks": 0.0,
        "laplacian_mae": 0.0,
        "laplacian_mae_blocks": 0.0,
        "highpass_mae": 0.0,
        "highpass_mae_blocks": 0.0,
        "support_mse": 0.0,
        "pred_roughness": 0.0,
        "target_roughness": 0.0,
        "context_style_error_blocks": 0.0,
        "context_roughness_ratio": 0.0,
        "context_laplacian_ratio": 0.0,
        "context_highpass_ratio": 0.0,
    }
    case_count = 0
    mask_pixel_count = 0
    seam_pixel_count = 0
    material_correct_count = 0
    material_pixel_count = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            batch = _build_validation_batch(case_dir, device, channels_last=channels_last)
            if batch is None:
                continue
            with torch.autocast(device_type=device.type, dtype=amp_config.dtype, enabled=amp_config.enabled):
                outputs = model(
                    known_height=batch["known_height"],
                    prefill_height=batch["prefill_height"],
                    mask=batch["mask"],
                    known_material=batch["known_material"],
                    known_support=batch["known_support"],
                    boundary_distance=batch["boundary_distance"],
                    prefill_gradients=batch["prefill_gradients"],
                    prefill_laplacian=batch["prefill_laplacian"],
                )
                predicted_height = batch["prefill_height"] + outputs.height_residual
                composite_height = batch["target_height"] * (1.0 - batch["mask"]) + predicted_height * batch["mask"]
                height_scale = height_scale_for_batch(batch, batch["target_height"])
                height_mae = masked_mean((composite_height - batch["target_height"]).abs(), batch["mask"])
                block_error = (composite_height - batch["target_height"]).abs() * height_scale
                height_mae_blocks = masked_mean(block_error, batch["mask"])
                height_within_1_block = masked_mean((block_error <= 1.0).float(), batch["mask"])
                height_within_2_blocks = masked_mean((block_error <= 2.0).float(), batch["mask"])

                pred_grad_x, pred_grad_y = height_gradients(composite_height)
                target_grad_x, target_grad_y = height_gradients(batch["target_height"])
                gradient_error = (pred_grad_x - target_grad_x).abs() + (pred_grad_y - target_grad_y).abs()
                gradient_mae = masked_mean(gradient_error, batch["mask"])
                gradient_mae_blocks = masked_mean(gradient_error * height_scale, batch["mask"])
                seam_mask = boundary_band(batch["mask"])
                seam_mae = masked_mean(gradient_error, seam_mask)
                seam_mae_blocks = masked_mean(gradient_error * height_scale, seam_mask)
                laplacian_error = (height_laplacian(composite_height) - height_laplacian(batch["target_height"])).abs()
                laplacian_mae = masked_mean(
                    laplacian_error,
                    batch["mask"],
                )
                laplacian_mae_blocks = masked_mean(laplacian_error * height_scale, batch["mask"])
                highpass_error = (height_highpass(composite_height) - height_highpass(batch["target_height"])).abs()
                highpass_mae = masked_mean(
                    highpass_error,
                    batch["mask"],
                )
                highpass_mae_blocks = masked_mean(highpass_error * height_scale, batch["mask"])
                pred_roughness = pred_grad_x.abs() + pred_grad_y.abs()
                target_roughness = target_grad_x.abs() + target_grad_y.abs()
                pred_laplacian = height_laplacian(composite_height).abs()
                target_laplacian = height_laplacian(batch["target_height"]).abs()
                pred_highpass = height_highpass(composite_height).abs()
                target_highpass = height_highpass(batch["target_height"]).abs()
                context_metrics = context_style_metrics(
                    (pred_roughness, pred_laplacian, pred_highpass),
                    (target_roughness, target_laplacian, target_highpass),
                    batch["mask"],
                    height_scale,
                )

                pred_material = outputs.material_logits.argmax(dim=1)
                material_mask = batch["mask"].squeeze(1).bool()
                material_correct = pred_material[material_mask] == batch["target_material"][material_mask]
                support_mse = masked_mean((outputs.support - batch["target_support"]) ** 2, batch["mask"])

            mask_pixels = int(batch["mask"].sum().detach().cpu().item())
            seam_pixels = int(seam_mask.sum().detach().cpu().item())
            material_pixels = int(material_mask.sum().detach().cpu().item())
            totals["height_mae"] += float(height_mae.detach().cpu()) * mask_pixels
            totals["height_mae_blocks"] += float(height_mae_blocks.detach().cpu()) * mask_pixels
            totals["height_within_1_block"] += float(height_within_1_block.detach().cpu()) * mask_pixels
            totals["height_within_2_blocks"] += float(height_within_2_blocks.detach().cpu()) * mask_pixels
            totals["gradient_mae"] += float(gradient_mae.detach().cpu()) * mask_pixels
            totals["gradient_mae_blocks"] += float(gradient_mae_blocks.detach().cpu()) * mask_pixels
            totals["laplacian_mae"] += float(laplacian_mae.detach().cpu()) * mask_pixels
            totals["laplacian_mae_blocks"] += float(laplacian_mae_blocks.detach().cpu()) * mask_pixels
            totals["highpass_mae"] += float(highpass_mae.detach().cpu()) * mask_pixels
            totals["highpass_mae_blocks"] += float(highpass_mae_blocks.detach().cpu()) * mask_pixels
            totals["seam_mae"] += float(seam_mae.detach().cpu()) * seam_pixels
            totals["seam_mae_blocks"] += float(seam_mae_blocks.detach().cpu()) * seam_pixels
            totals["support_mse"] += float(support_mse.detach().cpu()) * mask_pixels
            totals["pred_roughness"] += float(masked_sum(pred_roughness, batch["mask"]).detach().cpu())
            totals["target_roughness"] += float(masked_sum(target_roughness, batch["mask"]).detach().cpu())
            totals["context_style_error_blocks"] += float(context_metrics.style_error_blocks.detach().cpu()) * mask_pixels
            totals["context_roughness_ratio"] += float(context_metrics.roughness_ratio.detach().cpu()) * mask_pixels
            totals["context_laplacian_ratio"] += float(context_metrics.laplacian_ratio.detach().cpu()) * mask_pixels
            totals["context_highpass_ratio"] += float(context_metrics.highpass_ratio.detach().cpu()) * mask_pixels
            mask_pixel_count += mask_pixels
            seam_pixel_count += seam_pixels
            material_correct_count += int(material_correct.sum().detach().cpu().item())
            material_pixel_count += material_pixels
            case_count += 1

    if was_training:
        model.train()
    if case_count == 0:
        return None
    mask_denominator = max(mask_pixel_count, 1)
    seam_denominator = max(seam_pixel_count, 1)
    material_denominator = max(material_pixel_count, 1)
    averaged = {
        "height_mae": totals["height_mae"] / mask_denominator,
        "height_mae_blocks": totals["height_mae_blocks"] / mask_denominator,
        "height_within_1_block": totals["height_within_1_block"] / mask_denominator,
        "height_within_2_blocks": totals["height_within_2_blocks"] / mask_denominator,
        "seam_mae": totals["seam_mae"] / seam_denominator,
        "seam_mae_blocks": totals["seam_mae_blocks"] / seam_denominator,
        "gradient_mae": totals["gradient_mae"] / mask_denominator,
        "gradient_mae_blocks": totals["gradient_mae_blocks"] / mask_denominator,
        "laplacian_mae": totals["laplacian_mae"] / mask_denominator,
        "laplacian_mae_blocks": totals["laplacian_mae_blocks"] / mask_denominator,
        "highpass_mae": totals["highpass_mae"] / mask_denominator,
        "highpass_mae_blocks": totals["highpass_mae_blocks"] / mask_denominator,
        "material_accuracy": material_correct_count / material_denominator,
        "support_mse": totals["support_mse"] / mask_denominator,
        "context_style_error_blocks": totals["context_style_error_blocks"] / mask_denominator,
        "context_roughness_ratio": totals["context_roughness_ratio"] / mask_denominator,
        "context_laplacian_ratio": totals["context_laplacian_ratio"] / mask_denominator,
        "context_highpass_ratio": totals["context_highpass_ratio"] / mask_denominator,
    }
    roughness_ratio = totals["pred_roughness"] / max(totals["target_roughness"], 1e-8)
    roughness_penalty = abs(float(np.log(max(roughness_ratio, 1e-8))))
    visual_score = (
        0.45 * averaged["height_mae_blocks"]
        + 1.5 * averaged["seam_mae_blocks"]
        + 0.35 * averaged["gradient_mae_blocks"]
        + 0.15 * averaged["laplacian_mae_blocks"]
        + 0.15 * averaged["highpass_mae_blocks"]
        + 0.75 * averaged["context_style_error_blocks"]
        + 0.005 * roughness_penalty
        + 0.25 * (1.0 - averaged["height_within_2_blocks"])
        + 0.1 * averaged["support_mse"]
    )
    legacy_score = (
        averaged["height_mae"]
        + 0.5 * averaged["seam_mae"]
        + 0.2 * (1.0 - averaged["material_accuracy"])
        + 0.1 * averaged["support_mse"]
    )
    return RepairValidationMetrics(
        score=visual_score,
        visual_score=visual_score,
        legacy_score=legacy_score,
        height_mae=averaged["height_mae"],
        height_mae_blocks=averaged["height_mae_blocks"],
        height_within_1_block=averaged["height_within_1_block"],
        height_within_2_blocks=averaged["height_within_2_blocks"],
        seam_mae=averaged["seam_mae"],
        seam_mae_blocks=averaged["seam_mae_blocks"],
        gradient_mae=averaged["gradient_mae"],
        gradient_mae_blocks=averaged["gradient_mae_blocks"],
        laplacian_mae=averaged["laplacian_mae"],
        laplacian_mae_blocks=averaged["laplacian_mae_blocks"],
        highpass_mae=averaged["highpass_mae"],
        highpass_mae_blocks=averaged["highpass_mae_blocks"],
        roughness_ratio=roughness_ratio,
        context_style_error_blocks=averaged["context_style_error_blocks"],
        context_roughness_ratio=averaged["context_roughness_ratio"],
        context_laplacian_ratio=averaged["context_laplacian_ratio"],
        context_highpass_ratio=averaged["context_highpass_ratio"],
        material_accuracy=averaged["material_accuracy"],
        support_mse=averaged["support_mse"],
        case_count=case_count,
        mask_pixel_count=mask_pixel_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deterministic surface terrain repair.")
    parser.add_argument(
        "--export-dir",
        action="append",
        required=True,
        help="Export directory containing chunk/surface arrays. Repeat the flag or pass a comma-separated list to mix multiple exports.",
    )
    parser.add_argument("--checkpoint", required=True, help="Output checkpoint path")
    parser.add_argument("--latest-checkpoint", default=None, help="Optional latest-checkpoint path; defaults beside --checkpoint")
    parser.add_argument("--best-checkpoint", default=None, help="Optional best-checkpoint path; defaults beside --checkpoint")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=1, help="Total target epoch count, including resumed epochs")
    parser.add_argument("--save-every", type=int, default=1, help="Save a checkpoint every N epochs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0, help="Seed for training masks, shuffling, and model initialization.")
    parser.add_argument("--model-base-channels", type=int, default=64)
    parser.add_argument("--model-depth", type=int, default=4)
    parser.add_argument(
        "--model-bottleneck-dilations",
        default="1,2,4,2",
        help="Comma-separated dilation rates for bottleneck residual blocks. Empty disables extra dilated blocks.",
    )
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--stride-chunks", type=int, default=1)
    parser.add_argument(
        "--mask-mode",
        default="selection_mixed",
        choices=["none", "rectangle", "strip", "blob", "mixed", "terrain_mixed", "selection_mixed"],
    )
    parser.add_argument("--device", default="auto", help="Training device: auto, cuda, cuda:0, mps, or cpu")
    parser.add_argument("--amp", default="auto", choices=["auto", "off", "fp16", "bf16"], help="Mixed precision mode; auto enables CUDA AMP.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the training model when available.")
    parser.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last memory format for 4D floating tensors.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for feature preparation.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Clip gradient norm; set <= 0 to disable.")
    parser.add_argument("--matmul-precision", default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--tf32", default="auto", choices=["auto", "on", "off"], help="Allow TF32 matmul/cudnn on CUDA.")
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-cases-dir", default=None, help="Directory of fixed repair cases for epoch validation.")
    parser.add_argument("--validate-every", type=int, default=1, help="Validate every N epochs when validation cases are configured.")
    parser.add_argument("--tensorboard-dir", default=None, help="Write TensorBoard logs to this directory.")
    add_repair_loss_weight_args(parser)
    args = parser.parse_args()
    training_export_dirs = resolve_training_export_dirs(args.export_dir)
    configure_training_seed(args.seed)

    device = select_training_device(args.device)
    configure_cuda_backend(device, args.tf32, args.cudnn_benchmark)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)
    latest_checkpoint = Path(args.latest_checkpoint) if args.latest_checkpoint else checkpoint_sibling(args.checkpoint, "latest")
    best_checkpoint = Path(args.best_checkpoint) if args.best_checkpoint else checkpoint_sibling(args.checkpoint, "best")
    writer = create_summary_writer(args.tensorboard_dir)
    dataset = TerrainRepairDataset(
        training_export_dirs,
        tile_size=args.tile_size,
        stride_chunks=args.stride_chunks,
        mask_mode=args.mask_mode,
        seed=args.seed,
    )
    print_validation_overlap_warnings(args.validation_cases_dir, training_export_dirs)
    loader_kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    model = TerrainRepairUNet(
        num_material_classes=dataset.num_material_classes,
        base_channels=args.model_base_channels,
        depth=args.model_depth,
        bottleneck_dilations=args.model_bottleneck_dilations,
    ).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    weights = repair_loss_weights_from_args(args)
    amp_config = resolve_amp_config(device, args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_config.enabled and amp_config.dtype == torch.float16 and device.type == "cuda")
    train_model: TerrainRepairUNet | torch.nn.Module = model
    best_score = float("inf")

    state = RepairTrainingState()
    if args.resume is not None:
        payload = load_repair_checkpoint(args.resume, model, optimizer=optimizer, map_location=device)
        state = restore_repair_training_state(payload)
        meta = payload.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("best_score"), (int, float)):
            best_score = float(meta["best_score"])
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

    if args.compile:
        try:
            train_model = torch.compile(model, mode=args.compile_mode)
            print(f"Using torch.compile(mode={args.compile_mode})")
        except Exception as exc:
            train_model = model
            print(f"torch.compile unavailable; continuing without compile: {exc}")
    if amp_config.enabled:
        print(f"Using AMP dtype={amp_config.dtype}")
    if args.channels_last:
        print("Using channels-last memory format")
    print(f"Training on {len(dataset.export_dirs)} world{'s' if len(dataset.export_dirs) != 1 else ''}")
    print(
        f"Training on {device}; grad_accum_steps={max(1, args.grad_accum_steps)}; "
        f"exports={len(dataset.export_dirs)}; windows={len(dataset)}"
    )

    model.train()
    progress: tqdm | None = None
    try:
        for epoch in range(state.completed_epochs, args.epochs):
            dataset.set_mask_epoch(epoch)
            losses: RepairLossOutput | None = None
            progress = tqdm(loader, desc=f"Repair Epoch {epoch + 1}/{args.epochs}", unit="batch", dynamic_ncols=True)
            total_batches = len(loader)
            accum_steps = max(1, args.grad_accum_steps)
            for batch_index, batch in enumerate(progress):
                window_start = (batch_index // accum_steps) * accum_steps
                window_end = min(window_start + accum_steps, total_batches)
                window_size = window_end - window_start
                should_step = batch_index + 1 == window_end
                batch = move_repair_batch(batch, device, channels_last=args.channels_last)
                losses = train_repair_step(
                    train_model,
                    optimizer,
                    batch,
                    weights=weights,
                    amp_config=amp_config,
                    scaler=scaler,
                    grad_clip_norm=args.grad_clip_norm,
                    loss_scale=window_size,
                    step_optimizer=should_step,
                    zero_grad=batch_index == window_start,
                )
                if should_step:
                    state = RepairTrainingState(completed_epochs=epoch, global_step=state.global_step + 1)
                    if writer is not None and losses is not None:
                        writer.add_scalar("train/total_loss", losses.total_loss.item(), state.global_step)
                        writer.add_scalar("train/height_loss", losses.height_loss.item(), state.global_step)
                        writer.add_scalar("train/height_mae_blocks", losses.height_mae_blocks.item(), state.global_step)
                        writer.add_scalar("train/height_within_1_block", losses.height_within_1_block.item(), state.global_step)
                        writer.add_scalar("train/height_within_2_blocks", losses.height_within_2_blocks.item(), state.global_step)
                        writer.add_scalar("train/edge_height_loss", losses.edge_height_loss.item(), state.global_step)
                        writer.add_scalar("train/gradient_loss", losses.gradient_loss.item(), state.global_step)
                        writer.add_scalar("train/gradient_mae_blocks", losses.gradient_mae_blocks.item(), state.global_step)
                        writer.add_scalar("train/seam_loss", losses.seam_loss.item(), state.global_step)
                        writer.add_scalar("train/laplacian_loss", losses.laplacian_loss.item(), state.global_step)
                        writer.add_scalar("train/highpass_loss", losses.highpass_loss.item(), state.global_step)
                        writer.add_scalar("train/roughness_loss", losses.roughness_loss.item(), state.global_step)
                        writer.add_scalar("train/context_loss", losses.context_loss.item(), state.global_step)
                        writer.add_scalar("train/material_loss", losses.material_loss.item(), state.global_step)
                        writer.add_scalar("train/support_loss", losses.support_loss.item(), state.global_step)
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
                    f"height={losses.height_loss.item():.4f} height_blocks={losses.height_mae_blocks.item():.3f} "
                    f"within1={losses.height_within_1_block.item():.3f} within2={losses.height_within_2_blocks.item():.3f} "
                    f"edge={losses.edge_height_loss.item():.4f} "
                    f"gradient={losses.gradient_loss.item():.4f} seam={losses.seam_loss.item():.4f} "
                    f"laplacian={losses.laplacian_loss.item():.4f} highpass={losses.highpass_loss.item():.4f} "
                    f"roughness={losses.roughness_loss.item():.4f} context={losses.context_loss.item():.4f} "
                    f"material={losses.material_loss.item():.4f} "
                    f"support={losses.support_loss.item():.4f}"
                )
            validation_metrics = None
            if args.validation_cases_dir is not None and args.validate_every > 0 and ((epoch + 1) % args.validate_every == 0 or epoch + 1 == args.epochs):
                validation_metrics = evaluate_repair_cases(
                    model,
                    args.validation_cases_dir,
                    device=device,
                    amp_config=amp_config,
                    channels_last=args.channels_last,
                )
                if validation_metrics is not None:
                    print(
                        f"repair validation: visual_score={validation_metrics.visual_score:.4f} "
                        f"legacy_score={validation_metrics.legacy_score:.4f} "
                        f"height_blocks={validation_metrics.height_mae_blocks:.3f} "
                        f"within1={validation_metrics.height_within_1_block:.3f} "
                        f"within2={validation_metrics.height_within_2_blocks:.3f} "
                        f"seam_blocks={validation_metrics.seam_mae_blocks:.3f} "
                        f"grad_blocks={validation_metrics.gradient_mae_blocks:.3f} "
                        f"lap_blocks={validation_metrics.laplacian_mae_blocks:.3f} "
                        f"highpass_blocks={validation_metrics.highpass_mae_blocks:.3f} "
                        f"roughness_ratio={validation_metrics.roughness_ratio:.3f} "
                        f"context_blocks={validation_metrics.context_style_error_blocks:.3f} "
                        f"context_rough={validation_metrics.context_roughness_ratio:.3f} "
                        f"material_acc={validation_metrics.material_accuracy:.4f} support_mse={validation_metrics.support_mse:.4f} "
                        f"cases={validation_metrics.case_count} mask_pixels={validation_metrics.mask_pixel_count}"
                    )
                    if writer is not None:
                        writer.add_scalar("val/score", validation_metrics.score, state.global_step)
                        writer.add_scalar("val/visual_score", validation_metrics.visual_score, state.global_step)
                        writer.add_scalar("val/legacy_score", validation_metrics.legacy_score, state.global_step)
                        writer.add_scalar("val/height_mae", validation_metrics.height_mae, state.global_step)
                        writer.add_scalar("val/height_mae_blocks", validation_metrics.height_mae_blocks, state.global_step)
                        writer.add_scalar("val/height_within_1_block", validation_metrics.height_within_1_block, state.global_step)
                        writer.add_scalar("val/height_within_2_blocks", validation_metrics.height_within_2_blocks, state.global_step)
                        writer.add_scalar("val/seam_mae", validation_metrics.seam_mae, state.global_step)
                        writer.add_scalar("val/seam_mae_blocks", validation_metrics.seam_mae_blocks, state.global_step)
                        writer.add_scalar("val/gradient_mae", validation_metrics.gradient_mae, state.global_step)
                        writer.add_scalar("val/gradient_mae_blocks", validation_metrics.gradient_mae_blocks, state.global_step)
                        writer.add_scalar("val/laplacian_mae", validation_metrics.laplacian_mae, state.global_step)
                        writer.add_scalar("val/laplacian_mae_blocks", validation_metrics.laplacian_mae_blocks, state.global_step)
                        writer.add_scalar("val/highpass_mae", validation_metrics.highpass_mae, state.global_step)
                        writer.add_scalar("val/highpass_mae_blocks", validation_metrics.highpass_mae_blocks, state.global_step)
                        writer.add_scalar("val/roughness_ratio", validation_metrics.roughness_ratio, state.global_step)
                        writer.add_scalar("val/context_style_error_blocks", validation_metrics.context_style_error_blocks, state.global_step)
                        writer.add_scalar("val/context_roughness_ratio", validation_metrics.context_roughness_ratio, state.global_step)
                        writer.add_scalar("val/context_laplacian_ratio", validation_metrics.context_laplacian_ratio, state.global_step)
                        writer.add_scalar("val/context_highpass_ratio", validation_metrics.context_highpass_ratio, state.global_step)
                        writer.add_scalar("val/material_accuracy", validation_metrics.material_accuracy, state.global_step)
                        writer.add_scalar("val/support_mse", validation_metrics.support_mse, state.global_step)
                    if validation_metrics.score < best_score:
                        best_score = validation_metrics.score
                        best_args = argparse.Namespace(**vars(args), best_score=best_score)
                        persist_repair_checkpoint(best_checkpoint, model, optimizer, best_args, dataset, state, interrupted=False)
                        print(f"Saved best repair checkpoint to {best_checkpoint.resolve()}")
            if args.save_every > 0 and ((epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs):
                latest_args = argparse.Namespace(**vars(args), best_score=best_score if best_score < float("inf") else None)
                persist_repair_checkpoints(args.checkpoint, model, optimizer, latest_args, dataset, state, interrupted=False, latest_checkpoint_path=latest_checkpoint)
                print(f"Saved repair checkpoint to {Path(args.checkpoint).resolve()}")
    except KeyboardInterrupt:
        if progress is not None:
            progress.close()
        latest_args = argparse.Namespace(**vars(args), best_score=best_score if best_score < float("inf") else None)
        persist_repair_checkpoints(args.checkpoint, model, optimizer, latest_args, dataset, state, interrupted=True, latest_checkpoint_path=latest_checkpoint)
        print(f"Interrupted. Saved repair checkpoint to {Path(args.checkpoint).resolve()}")
        if writer is not None:
            writer.close()
        return

    if args.save_every <= 0:
        latest_args = argparse.Namespace(**vars(args), best_score=best_score if best_score < float("inf") else None)
        persist_repair_checkpoints(args.checkpoint, model, optimizer, latest_args, dataset, state, interrupted=False, latest_checkpoint_path=latest_checkpoint)
        print(f"Saved repair checkpoint to {Path(args.checkpoint).resolve()}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()


__all__ = [
    "RepairLossOutput",
    "RepairLossWeights",
    "RepairTrainingState",
    "add_repair_loss_weight_args",
    "build_repair_checkpoint_meta",
    "charbonnier",
    "configure_training_seed",
    "context_band",
    "context_style_loss",
    "compute_repair_losses",
    "height_highpass",
    "height_laplacian",
    "height_scale_for_batch",
    "load_repair_checkpoint",
    "load_repair_model_from_checkpoint",
    "move_repair_batch",
    "print_validation_overlap_warnings",
    "repair_loss_weights_from_args",
    "resolve_amp_config",
    "restore_repair_training_state",
    "save_repair_checkpoint",
    "select_training_device",
    "train_repair_step",
    "validation_overlap_warnings",
]
