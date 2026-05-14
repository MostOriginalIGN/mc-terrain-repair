"""Helpers for building diffusion inference inputs from exported terrain chunks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .data import CHUNK_SIZE, TerrainDiffusionDataset


@dataclass(frozen=True)
class SelectionPlan:
    origin_chunk_x: int
    origin_chunk_z: int
    selected_min_chunk_x: int
    selected_min_chunk_z: int
    selected_max_chunk_x: int
    selected_max_chunk_z: int
    mask_top: int
    mask_left: int
    mask_height: int
    mask_width: int


def load_height_range(checkpoint_path: str | Path | None) -> tuple[float, float] | None:
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


def find_origin(dataset: TerrainDiffusionDataset, origin_x: int | None, origin_z: int | None) -> tuple[int, int]:
    if origin_x is None and origin_z is None:
        return dataset.window_origins[0]
    if origin_x is None or origin_z is None:
        raise SystemExit('Provide both --origin-chunk-x and --origin-chunk-z, or neither.')
    origin = (origin_x, origin_z)
    if origin not in dataset.window_origins:
        raise SystemExit(f'No contiguous window found at chunk origin {origin}.')
    return origin


def plan_chunk_selection(
    window_origins: list[tuple[int, int]],
    chunks_per_side: int,
    selected_min_chunk_x: int,
    selected_min_chunk_z: int,
    selected_max_chunk_x: int,
    selected_max_chunk_z: int,
) -> SelectionPlan:
    width_chunks = selected_max_chunk_x - selected_min_chunk_x + 1
    height_chunks = selected_max_chunk_z - selected_min_chunk_z + 1
    if width_chunks <= 0 or height_chunks <= 0:
        raise ValueError('Selection must cover at least one chunk.')
    if width_chunks > chunks_per_side or height_chunks > chunks_per_side:
        raise ValueError(
            f'Selection is {width_chunks}x{height_chunks} chunks but tile_size only supports '
            f'{chunks_per_side}x{chunks_per_side} chunks.'
        )

    candidate_origins = [
        (origin_x, origin_z)
        for origin_x, origin_z in window_origins
        if origin_x <= selected_min_chunk_x
        and origin_z <= selected_min_chunk_z
        and origin_x + chunks_per_side - 1 >= selected_max_chunk_x
        and origin_z + chunks_per_side - 1 >= selected_max_chunk_z
    ]
    if not candidate_origins:
        raise ValueError('No contiguous inference window contains the selected chunks.')

    selected_center_x = (selected_min_chunk_x + selected_max_chunk_x) / 2.0
    selected_center_z = (selected_min_chunk_z + selected_max_chunk_z) / 2.0

    def score(origin: tuple[int, int]) -> tuple[float, int, int]:
        origin_x, origin_z = origin
        window_center_x = origin_x + (chunks_per_side - 1) / 2.0
        window_center_z = origin_z + (chunks_per_side - 1) / 2.0
        distance = abs(window_center_x - selected_center_x) + abs(window_center_z - selected_center_z)
        return (distance, origin_z, origin_x)

    origin_chunk_x, origin_chunk_z = min(candidate_origins, key=score)
    mask_left = (selected_min_chunk_x - origin_chunk_x) * CHUNK_SIZE
    mask_top = (selected_min_chunk_z - origin_chunk_z) * CHUNK_SIZE
    mask_width = width_chunks * CHUNK_SIZE
    mask_height = height_chunks * CHUNK_SIZE

    return SelectionPlan(
        origin_chunk_x=origin_chunk_x,
        origin_chunk_z=origin_chunk_z,
        selected_min_chunk_x=selected_min_chunk_x,
        selected_min_chunk_z=selected_min_chunk_z,
        selected_max_chunk_x=selected_max_chunk_x,
        selected_max_chunk_z=selected_max_chunk_z,
        mask_top=mask_top,
        mask_left=mask_left,
        mask_height=mask_height,
        mask_width=mask_width,
    )


def prepare_inference_inputs(
    export_dir: str | Path,
    out_dir: str | Path,
    checkpoint: str | Path | None = None,
    tile_size: int = 128,
    origin_chunk_x: int | None = None,
    origin_chunk_z: int | None = None,
    mask_top: int = 48,
    mask_left: int = 48,
    mask_height: int = 32,
    mask_width: int = 32,
) -> dict[str, object]:
    export_path = Path(export_dir).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    height_range = load_height_range(checkpoint)
    dataset = TerrainDiffusionDataset(
        export_path,
        tile_size=tile_size,
        mask_mode='none',
        height_range=height_range,
    )
    resolved_origin_x, resolved_origin_z = find_origin(dataset, origin_chunk_x, origin_chunk_z)

    surface = dataset._assemble_surface_window(resolved_origin_x, resolved_origin_z)
    target_height = dataset._normalize_height(surface)
    target_material = dataset._assemble_material_window(resolved_origin_x, resolved_origin_z)

    mask = np.zeros((tile_size, tile_size), dtype=np.float32)
    bottom = mask_top + mask_height
    right = mask_left + mask_width
    if mask_top < 0 or mask_left < 0 or bottom > tile_size or right > tile_size:
        raise SystemExit('Mask rectangle must fit inside the chosen tile size.')
    mask[mask_top:bottom, mask_left:right] = 1.0

    known_height = target_height * (1.0 - mask)
    known_material = target_material.copy()
    known_material[mask.astype(bool)] = 0

    np.save(out_path / 'known_height.npy', known_height.astype(np.float32))
    np.save(out_path / 'known_material.npy', known_material.astype(np.int64))
    np.save(out_path / 'mask.npy', mask.astype(np.float32))
    np.save(out_path / 'target_height.npy', target_height.astype(np.float32))
    np.save(out_path / 'target_material.npy', target_material.astype(np.int64))

    metadata = {
        'origin_chunk_x': resolved_origin_x,
        'origin_chunk_z': resolved_origin_z,
        'tile_size': tile_size,
        'mask_top': mask_top,
        'mask_left': mask_left,
        'mask_height': mask_height,
        'mask_width': mask_width,
        'height_min': dataset.height_min,
        'height_max': dataset.height_max,
        'export_dir': str(export_path),
    }
    (out_path / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata
