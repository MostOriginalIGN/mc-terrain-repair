"""Shared terrain tile dataset assembly for Minecraft repair training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from exporter.vocab import NUM_CLASSES

CHUNK_SIZE = 16
SURFACE_INDEX = 32
SURFACE_FILENAME_RE = re.compile(r"^surface_(-?\d+)_(-?\d+)\.npy$")
CHUNK_FILENAME_RE = re.compile(r"^chunk_(-?\d+)_(-?\d+)\.npy$")


@dataclass(frozen=True)
class TerrainWindowSample:
    target_height: torch.Tensor
    target_material: torch.Tensor
    known_height: torch.Tensor
    known_material: torch.Tensor
    mask: torch.Tensor
    gradients: torch.Tensor
    origin_chunk_x: int
    origin_chunk_z: int


class TerrainDiffusionDataset(Dataset[dict[str, torch.Tensor]]):
    """Assemble contiguous terrain windows on the fly from exported chunks."""

    def __init__(
        self,
        export_dir: str | Path,
        tile_size: int = 128,
        stride_chunks: int = 1,
        mask_mode: str = "rectangle",
        mask_fraction_range: tuple[float, float] = (0.15, 0.5),
        seed: int = 0,
        cache_arrays: bool = True,
        height_range: tuple[float, float] | None = None,
    ):
        if tile_size % CHUNK_SIZE != 0:
            raise ValueError(f"tile_size must be divisible by {CHUNK_SIZE}, got {tile_size}")
        self.export_dir = Path(export_dir)
        self.tile_size = tile_size
        self.chunks_per_side = tile_size // CHUNK_SIZE
        self.stride_chunks = stride_chunks
        self.mask_mode = mask_mode
        self.mask_fraction_range = mask_fraction_range
        self.seed = seed
        self.cache_arrays = cache_arrays
        self.surface_paths, self.chunk_paths = self._index_exports(self.export_dir)
        self.window_origins = self._discover_windows()
        if not self.window_origins:
            raise ValueError(
                f"No contiguous {self.chunks_per_side}x{self.chunks_per_side} chunk windows found in {self.export_dir}"
            )
        self._surface_cache: dict[tuple[int, int], np.ndarray] = {}
        self._chunk_cache: dict[tuple[int, int], np.ndarray] = {}
        self.height_min, self.height_max = height_range or self._compute_height_range()
        self.num_material_classes = NUM_CLASSES

    def __len__(self) -> int:
        return len(self.window_origins)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        origin_x, origin_z = self.window_origins[index]
        surface = self._assemble_surface_window(origin_x, origin_z)
        materials = self._assemble_material_window(origin_x, origin_z)

        target_height = self._normalize_height(surface)
        mask = self._build_mask(index)
        known_height = target_height * (1.0 - mask)
        known_material = materials.copy()
        known_material[mask.astype(bool)] = 0
        gradients = self._compute_gradients(known_height)

        sample = TerrainWindowSample(
            target_height=torch.from_numpy(target_height[None, ...]).float(),
            target_material=torch.from_numpy(materials).long(),
            known_height=torch.from_numpy(known_height[None, ...]).float(),
            known_material=torch.from_numpy(known_material).long(),
            mask=torch.from_numpy(mask[None, ...]).float(),
            gradients=torch.from_numpy(gradients).float(),
            origin_chunk_x=origin_x,
            origin_chunk_z=origin_z,
        )
        return {
            "target_height": sample.target_height,
            "target_material": sample.target_material,
            "known_height": sample.known_height,
            "known_material": sample.known_material,
            "mask": sample.mask,
            "gradients": sample.gradients,
            "origin_chunk_x": torch.tensor(sample.origin_chunk_x),
            "origin_chunk_z": torch.tensor(sample.origin_chunk_z),
        }

    def denormalize_height(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * (self.height_max - self.height_min) + self.height_min

    def _index_exports(self, export_dir: Path) -> tuple[dict[tuple[int, int], Path], dict[tuple[int, int], Path]]:
        surface_paths: dict[tuple[int, int], Path] = {}
        chunk_paths: dict[tuple[int, int], Path] = {}
        for path in export_dir.glob('surface_*.npy'):
            match = SURFACE_FILENAME_RE.match(path.name)
            if match:
                surface_paths[(int(match.group(1)), int(match.group(2)))] = path
        for path in export_dir.glob('chunk_*.npy'):
            match = CHUNK_FILENAME_RE.match(path.name)
            if match:
                chunk_paths[(int(match.group(1)), int(match.group(2)))] = path
        shared = sorted(set(surface_paths) & set(chunk_paths))
        return ({coord: surface_paths[coord] for coord in shared}, {coord: chunk_paths[coord] for coord in shared})

    def _discover_windows(self) -> list[tuple[int, int]]:
        coords = set(self.surface_paths)
        xs = sorted({x for x, _ in coords})
        zs = sorted({z for _, z in coords})
        if not xs or not zs:
            return []
        origins: list[tuple[int, int]] = []
        min_x, max_x = xs[0], xs[-1]
        min_z, max_z = zs[0], zs[-1]
        for origin_x in range(min_x, max_x - self.chunks_per_side + 2, self.stride_chunks):
            for origin_z in range(min_z, max_z - self.chunks_per_side + 2, self.stride_chunks):
                if all(
                    (origin_x + dx, origin_z + dz) in coords
                    for dx in range(self.chunks_per_side)
                    for dz in range(self.chunks_per_side)
                ):
                    origins.append((origin_x, origin_z))
        return origins

    def _compute_height_range(self) -> tuple[float, float]:
        min_height = float('inf')
        max_height = float('-inf')
        for coord in self.surface_paths:
            surface = self._load_surface(coord)
            min_height = min(min_height, float(surface.min()))
            max_height = max(max_height, float(surface.max()))
        if min_height == max_height:
            max_height = min_height + 1.0
        return min_height, max_height

    def _load_surface(self, coord: tuple[int, int]) -> np.ndarray:
        if self.cache_arrays and coord in self._surface_cache:
            return self._surface_cache[coord]
        surface = np.load(self.surface_paths[coord])
        if self.cache_arrays:
            self._surface_cache[coord] = surface
        return surface

    def _load_chunk(self, coord: tuple[int, int]) -> np.ndarray:
        if self.cache_arrays and coord in self._chunk_cache:
            return self._chunk_cache[coord]
        chunk = np.load(self.chunk_paths[coord])
        if self.cache_arrays:
            self._chunk_cache[coord] = chunk
        return chunk

    def _assemble_surface_window(self, origin_x: int, origin_z: int) -> np.ndarray:
        window = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        for dx in range(self.chunks_per_side):
            for dz in range(self.chunks_per_side):
                coord = (origin_x + dx, origin_z + dz)
                tile = self._load_surface(coord).T.astype(np.float32)
                row = dz * CHUNK_SIZE
                col = dx * CHUNK_SIZE
                window[row:row + CHUNK_SIZE, col:col + CHUNK_SIZE] = tile
        return window

    def _assemble_material_window(self, origin_x: int, origin_z: int) -> np.ndarray:
        window = np.zeros((self.tile_size, self.tile_size), dtype=np.int64)
        for dx in range(self.chunks_per_side):
            for dz in range(self.chunks_per_side):
                coord = (origin_x + dx, origin_z + dz)
                tile = self._load_chunk(coord)[:, :, SURFACE_INDEX].T.astype(np.int64)
                row = dz * CHUNK_SIZE
                col = dx * CHUNK_SIZE
                window[row:row + CHUNK_SIZE, col:col + CHUNK_SIZE] = tile
        return window

    def _normalize_height(self, surface: np.ndarray) -> np.ndarray:
        return (surface - self.height_min) / (self.height_max - self.height_min)

    def _build_mask(self, index: int) -> np.ndarray:
        if self.mask_mode == 'none':
            return np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        if self.mask_mode != 'rectangle':
            raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")

        rng = np.random.default_rng(self.seed + index)
        min_frac, max_frac = self.mask_fraction_range
        min_size = max(4, int(self.tile_size * min_frac))
        max_size = max(min_size, int(self.tile_size * max_frac))
        height = int(rng.integers(min_size, max_size + 1))
        width = int(rng.integers(min_size, max_size + 1))
        top = int(rng.integers(0, self.tile_size - height + 1))
        left = int(rng.integers(0, self.tile_size - width + 1))
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        mask[top:top + height, left:left + width] = 1.0
        return mask

    def _compute_gradients(self, known_height: np.ndarray) -> np.ndarray:
        grad_y, grad_x = np.gradient(known_height.astype(np.float32))
        return np.stack([grad_x, grad_y], axis=0)
