"""Training utilities for terrain diffusion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import TerrainDiffusionDataset
from .model import TerrainDiffusionUNet
from .scheduler import GaussianDiffusionScheduler


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    noise_loss: torch.Tensor
    material_loss: torch.Tensor


def compute_losses(
    model: TerrainDiffusionUNet,
    scheduler: GaussianDiffusionScheduler,
    batch: dict[str, torch.Tensor],
    material_loss_weight: float = 0.1,
) -> LossOutput:
    target_height = batch['target_height']
    target_material = batch['target_material']
    known_height = batch['known_height']
    known_material = batch['known_material']
    mask = batch['mask']
    gradients = batch['gradients']

    timesteps = scheduler.sample_timesteps(target_height.shape[0], target_height.device)
    noise = torch.randn_like(target_height)
    noisy_height = scheduler.q_sample(target_height, timesteps, noise)
    outputs = model(noisy_height, known_height, mask, known_material, gradients, timesteps)

    mask_weight = mask
    if mask_weight.sum() == 0:
        mask_weight = torch.ones_like(mask)
    noise_loss = ((outputs.noise_pred - noise) ** 2 * mask_weight).sum() / mask_weight.sum().clamp(min=1.0)

    material_loss_map = nn.functional.cross_entropy(outputs.material_logits, target_material, reduction='none')
    material_mask = mask.squeeze(1)
    if material_mask.sum() == 0:
        material_mask = torch.ones_like(material_mask)
    material_loss = (material_loss_map * material_mask).sum() / material_mask.sum().clamp(min=1.0)

    total_loss = noise_loss + material_loss_weight * material_loss
    return LossOutput(total_loss=total_loss, noise_loss=noise_loss, material_loss=material_loss)


def train_step(
    model: TerrainDiffusionUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: GaussianDiffusionScheduler,
    batch: dict[str, torch.Tensor],
    material_loss_weight: float = 0.1,
) -> LossOutput:
    optimizer.zero_grad(set_to_none=True)
    losses = compute_losses(model, scheduler, batch, material_loss_weight=material_loss_weight)
    losses.total_loss.backward()
    optimizer.step()
    return losses


def save_checkpoint(
    path: str | Path,
    model: TerrainDiffusionUNet,
    optimizer: torch.optim.Optimizer | None,
    scheduler: GaussianDiffusionScheduler,
    meta: dict[str, object] | None = None,
) -> None:
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_config': scheduler.config.__dict__,
        'num_material_classes': model.num_material_classes,
        'meta': meta or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: TerrainDiffusionUNet,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = 'cpu',
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None and checkpoint.get('optimizer_state') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the surface-oriented terrain diffusion scaffold.')
    parser.add_argument('--export-dir', required=True, help='Directory containing exported chunk and surface arrays')
    parser.add_argument('--checkpoint', required=True, help='Output checkpoint path')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--tile-size', type=int, default=128)
    parser.add_argument('--stride-chunks', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TerrainDiffusionDataset(
        args.export_dir,
        tile_size=args.tile_size,
        stride_chunks=args.stride_chunks,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = TerrainDiffusionUNet(num_material_classes=dataset.num_material_classes).to(device)
    scheduler = GaussianDiffusionScheduler().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        losses: LossOutput | None = None
        progress = tqdm(
            loader,
            desc=f'Epoch {epoch + 1}/{args.epochs}',
            unit='batch',
            dynamic_ncols=True,
        )
        for batch in progress:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            losses = train_step(model, optimizer, scheduler, batch)
            progress.set_postfix({
                'total': f'{losses.total_loss.item():.4f}',
                'noise': f'{losses.noise_loss.item():.4f}',
                'material': f'{losses.material_loss.item():.4f}',
            })
        progress.close()
        if losses is not None:
            print(
                f"epoch {epoch + 1}/{args.epochs}: "
                f"total={losses.total_loss.item():.4f} noise={losses.noise_loss.item():.4f} material={losses.material_loss.item():.4f}"
            )

    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        scheduler,
        meta={
            'tile_size': args.tile_size,
            'stride_chunks': args.stride_chunks,
            'height_min': dataset.height_min,
            'height_max': dataset.height_max,
            'export_dir': str(Path(args.export_dir).resolve()),
        },
    )
    print(f"Saved checkpoint to {Path(args.checkpoint).resolve()}")


if __name__ == '__main__':
    main()
