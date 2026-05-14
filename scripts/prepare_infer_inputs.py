"""Prepare masked inference inputs from exported terrain arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_SRC = ROOT / 'packages' / 'diffusion' / 'src'
EXPORTER_SRC = ROOT / 'packages' / 'exporter' / 'src'
DATASET_SRC = ROOT / 'packages' / 'dataset' / 'src'

for src_path in (str(DIFFUSION_SRC), str(EXPORTER_SRC), str(DATASET_SRC)):
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from diffusion.data import TerrainDiffusionDataset


def _load_height_range(checkpoint_path: str | None) -> tuple[float, float] | None:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f'Checkpoint does not exist: {path}')
    payload = torch.load(path, map_location='cpu')
    meta = payload.get('meta')
    if not isinstance(meta, dict):
        return None
    height_min = meta.get('height_min')
    height_max = meta.get('height_max')
    if isinstance(height_min, (int, float)) and isinstance(height_max, (int, float)):
        return float(height_min), float(height_max)
    return None


def _find_origin(dataset: TerrainDiffusionDataset, origin_x: int | None, origin_z: int | None) -> tuple[int, int]:
    if origin_x is None and origin_z is None:
        return dataset.window_origins[0]
    if origin_x is None or origin_z is None:
        raise SystemExit('Provide both --origin-chunk-x and --origin-chunk-z, or neither.')
    origin = (origin_x, origin_z)
    if origin not in dataset.window_origins:
        raise SystemExit(f'No contiguous window found at chunk origin {origin}.')
    return origin


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare known_height, known_material, and mask arrays for diffusion inference.')
    parser.add_argument('--export-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--checkpoint', default=None, help='Optional checkpoint to reuse saved height normalization range')
    parser.add_argument('--tile-size', type=int, default=128)
    parser.add_argument('--origin-chunk-x', type=int, default=None)
    parser.add_argument('--origin-chunk-z', type=int, default=None)
    parser.add_argument('--mask-top', type=int, default=48)
    parser.add_argument('--mask-left', type=int, default=48)
    parser.add_argument('--mask-height', type=int, default=32)
    parser.add_argument('--mask-width', type=int, default=32)
    args = parser.parse_args()

    export_dir = Path(args.export_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    height_range = _load_height_range(args.checkpoint)
    dataset = TerrainDiffusionDataset(
        export_dir,
        tile_size=args.tile_size,
        mask_mode='none',
        height_range=height_range,
    )
    origin_x, origin_z = _find_origin(dataset, args.origin_chunk_x, args.origin_chunk_z)

    surface = dataset._assemble_surface_window(origin_x, origin_z)
    target_height = dataset._normalize_height(surface)
    target_material = dataset._assemble_material_window(origin_x, origin_z)

    mask = np.zeros((args.tile_size, args.tile_size), dtype=np.float32)
    bottom = args.mask_top + args.mask_height
    right = args.mask_left + args.mask_width
    if args.mask_top < 0 or args.mask_left < 0 or bottom > args.tile_size or right > args.tile_size:
        raise SystemExit('Mask rectangle must fit inside the chosen tile size.')
    mask[args.mask_top:bottom, args.mask_left:right] = 1.0

    known_height = target_height * (1.0 - mask)
    known_material = target_material.copy()
    known_material[mask.astype(bool)] = 0

    np.save(out_dir / 'known_height.npy', known_height.astype(np.float32))
    np.save(out_dir / 'known_material.npy', known_material.astype(np.int64))
    np.save(out_dir / 'mask.npy', mask.astype(np.float32))
    np.save(out_dir / 'target_height.npy', target_height.astype(np.float32))
    np.save(out_dir / 'target_material.npy', target_material.astype(np.int64))

    metadata = {
        'origin_chunk_x': origin_x,
        'origin_chunk_z': origin_z,
        'tile_size': args.tile_size,
        'mask_top': args.mask_top,
        'mask_left': args.mask_left,
        'mask_height': args.mask_height,
        'mask_width': args.mask_width,
        'height_min': dataset.height_min,
        'height_max': dataset.height_max,
        'export_dir': str(export_dir),
    }
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared inference inputs in {out_dir}')


if __name__ == '__main__':
    main()
