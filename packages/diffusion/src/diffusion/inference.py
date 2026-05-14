"""MultiDiffusion-style tiled inference for terrain repair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from exporter.visualize import render_heightmap
from exporter.vocab import NUM_CLASSES

from .model import TerrainDiffusionUNet
from .scheduler import GaussianDiffusionScheduler
from .training import load_checkpoint


def _tile_starts(size: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size > size:
        return [0]
    step = max(1, tile_size - overlap)
    starts = list(range(0, size - tile_size + 1, step))
    if starts[-1] != size - tile_size:
        starts.append(size - tile_size)
    return starts


def _iter_tiles(height: int, width: int, tile_size: int, overlap: int):
    for top in _tile_starts(height, tile_size, overlap):
        for left in _tile_starts(width, tile_size, overlap):
            yield top, left


def _weight_map(tile_size: int, overlap: int, device: torch.device) -> torch.Tensor:
    if overlap <= 0:
        return torch.ones(1, 1, tile_size, tile_size, device=device)
    ramp = torch.ones(tile_size, device=device)
    edge = min(overlap, tile_size // 2)
    if edge > 0:
        vals = torch.linspace(0.1, 1.0, edge, device=device)
        ramp[:edge] = vals
        ramp[-edge:] = vals.flip(0)
    weight = torch.outer(ramp, ramp)
    return weight.unsqueeze(0).unsqueeze(0)


def _compute_gradients(known_height: torch.Tensor) -> torch.Tensor:
    height_2d = known_height.squeeze(1)
    grad_y, grad_x = torch.gradient(height_2d, dim=(1, 2))
    return torch.stack([grad_x, grad_y], dim=1)


def _load_array(path_str: str, name: str) -> np.ndarray:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(
            f"Missing {name} file: {path}. Use scripts/prepare_infer_inputs.py or `make prepare-infer` first."
        )
    return np.load(path)


def _maybe_denormalize(height: np.ndarray, checkpoint_payload: dict[str, object]) -> np.ndarray | None:
    meta = checkpoint_payload.get('meta')
    if not isinstance(meta, dict):
        return None
    height_min = meta.get('height_min')
    height_max = meta.get('height_max')
    if not isinstance(height_min, (int, float)) or not isinstance(height_max, (int, float)):
        return None
    return height * (float(height_max) - float(height_min)) + float(height_min)


def multidiffusion_inpaint(
    model: TerrainDiffusionUNet,
    scheduler: GaussianDiffusionScheduler,
    known_height: torch.Tensor,
    known_material: torch.Tensor,
    mask: torch.Tensor,
    gradients: torch.Tensor | None = None,
    tile_size: int = 128,
    overlap: int = 32,
    num_steps: int | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    if known_height.ndim != 4 or known_height.shape[1] != 1:
        raise ValueError(f"known_height must have shape [B, 1, H, W], got {tuple(known_height.shape)}")
    if mask.shape != known_height.shape:
        raise ValueError('mask must match known_height shape')
    if known_material.ndim != 3:
        raise ValueError('known_material must have shape [B, H, W]')

    device = known_height.device
    gradients = gradients if gradients is not None else _compute_gradients(known_height)
    latents = torch.randn_like(known_height, generator=generator)
    height, width = known_height.shape[-2:]
    weight = _weight_map(tile_size, overlap, device)
    timesteps = scheduler.inference_timesteps(device)
    if num_steps is not None:
        timesteps = timesteps[:num_steps]

    last_logits = None
    model.eval()
    step_bar = tqdm(timesteps, desc='Inference', unit='step', dynamic_ncols=True)
    with torch.no_grad():
        for timestep in step_bar:
            t = torch.full((known_height.shape[0],), int(timestep.item()), device=device, dtype=torch.long)
            accum_noise = torch.zeros_like(latents)
            accum_logits = torch.zeros(
                known_height.shape[0],
                model.num_material_classes,
                height,
                width,
                device=device,
            )
            accum_weight = torch.zeros_like(latents)

            for top, left in _iter_tiles(height, width, tile_size, overlap):
                row = slice(top, top + tile_size)
                col = slice(left, left + tile_size)
                outputs = model(
                    latents[:, :, row, col],
                    known_height[:, :, row, col],
                    mask[:, :, row, col],
                    known_material[:, row, col],
                    gradients[:, :, row, col],
                    t,
                )
                accum_noise[:, :, row, col] += outputs.noise_pred * weight
                accum_logits[:, :, row, col] += outputs.material_logits * weight
                accum_weight[:, :, row, col] += weight

            mean_noise = accum_noise / accum_weight.clamp(min=1e-6)
            last_logits = accum_logits / accum_weight.clamp(min=1e-6)
            latents = scheduler.step(mean_noise, t, latents)
            latents = known_height * (1.0 - mask) + latents * mask
        step_bar.close()

    material_logits = last_logits if last_logits is not None else torch.zeros(
        known_height.shape[0], model.num_material_classes, height, width, device=device
    )
    return {
        'height': latents,
        'material_logits': material_logits,
        'material': material_logits.argmax(dim=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run tiled terrain diffusion inference.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--known-height', required=True)
    parser.add_argument('--known-material', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--tile-size', type=int, default=128)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--num-steps', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TerrainDiffusionUNet(num_material_classes=NUM_CLASSES).to(device)
    checkpoint_payload = load_checkpoint(args.checkpoint, model, map_location=device)
    scheduler = GaussianDiffusionScheduler().to(device)

    known_height_array = _load_array(args.known_height, 'known height')
    known_material_array = _load_array(args.known_material, 'known material')
    mask_array = _load_array(args.mask, 'mask')

    known_height = torch.from_numpy(known_height_array).float().unsqueeze(0).unsqueeze(0).to(device)
    known_material = torch.from_numpy(known_material_array).long().unsqueeze(0).to(device)
    mask = torch.from_numpy(mask_array).float().unsqueeze(0).unsqueeze(0).to(device)

    result = multidiffusion_inpaint(
        model=model,
        scheduler=scheduler,
        known_height=known_height,
        known_material=known_material,
        mask=mask,
        tile_size=args.tile_size,
        overlap=args.overlap,
        num_steps=args.num_steps,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_height = result['height'].squeeze(0).squeeze(0).cpu().numpy()
    np.save(out_dir / 'height.npy', normalized_height)
    np.save(out_dir / 'material.npy', result['material'].squeeze(0).cpu().numpy())
    world_height = _maybe_denormalize(normalized_height, checkpoint_payload)
    preview_height = world_height if world_height is not None else normalized_height
    if world_height is not None:
        np.save(out_dir / 'height_world.npy', world_height)
    render_heightmap(preview_height, out_dir / 'height_preview.png', mask=mask_array, upscale=4)
    print(f"Saved inference outputs to {out_dir.resolve()}")


if __name__ == '__main__':
    main()
