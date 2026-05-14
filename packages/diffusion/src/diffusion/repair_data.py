"""Dataset and feature utilities for deterministic terrain repair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from exporter.vocab import AIR_INDEX, UNKNOWN_INDEX

from .data import CHUNK_SIZE, SURFACE_INDEX, TerrainDiffusionDataset

WATER_INDEX = 11


@dataclass(frozen=True)
class TerrainRepairSample:
    target_height: torch.Tensor
    target_material: torch.Tensor
    target_support: torch.Tensor
    known_height: torch.Tensor
    known_material: torch.Tensor
    known_support: torch.Tensor
    mask: torch.Tensor
    prefill_height: torch.Tensor
    boundary_distance: torch.Tensor
    prefill_gradients: torch.Tensor
    prefill_laplacian: torch.Tensor
    origin_chunk_x: int
    origin_chunk_z: int


def solid_mask(blocks: np.ndarray) -> np.ndarray:
    """Return terrain-like solidity for encoded block arrays."""
    return (
        (blocks != AIR_INDEX)
        & (blocks != WATER_INDEX)
        & (blocks != UNKNOWN_INDEX)
    )


def compute_support_from_chunk(chunk: np.ndarray) -> np.ndarray:
    """Approximate overhang support from the slab below the exported surface anchor."""
    if chunk.ndim != 3 or chunk.shape[2] <= SURFACE_INDEX:
        raise ValueError(f"Expected chunk slab shape [16, 16, >= {SURFACE_INDEX + 1}], got {chunk.shape}")
    below_surface = chunk[:, :, :SURFACE_INDEX]
    return solid_mask(below_surface).mean(axis=2).astype(np.float32)


def estimate_support_from_material(material: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Fallback support proxy when only a surface material map is available."""
    support = solid_mask(material).astype(np.float32)
    if mask is not None:
        support = support * (1.0 - mask.astype(np.float32))
    return support


def build_prefill_height(target_height: np.ndarray, mask: np.ndarray, iterations: int = 64) -> np.ndarray:
    """Fill masked height values by iterative neighbor averaging while preserving known pixels."""
    if target_height.shape != mask.shape:
        raise ValueError(f"target_height shape {target_height.shape} must match mask shape {mask.shape}")
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return target_height.astype(np.float32).copy()

    known = ~mask_bool
    if known.any():
        fill_value = float(target_height[known].mean())
    else:
        fill_value = 0.5
    filled = target_height.astype(np.float32).copy()
    filled[mask_bool] = fill_value

    for _ in range(max(1, iterations)):
        padded = np.pad(filled, 1, mode="edge")
        neighbor_avg = (
            padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
        ) * 0.25
        filled[mask_bool] = neighbor_avg[mask_bool]
        filled[known] = target_height[known]
    return filled.clip(0.0, 1.0).astype(np.float32)


def compute_boundary_distance(mask: np.ndarray) -> np.ndarray:
    """Compute a normalized city-block distance from masked cells to known terrain."""
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return np.zeros(mask.shape, dtype=np.float32)
    if mask_bool.all():
        return np.ones(mask.shape, dtype=np.float32)

    height, width = mask.shape
    dist = np.where(mask_bool, np.inf, 0.0).astype(np.float32)
    for _ in range(height + width):
        old = dist.copy()
        padded = np.pad(dist, 1, mode="constant", constant_values=np.inf)
        dist = np.minimum(
            dist,
            np.minimum.reduce([
                padded[1:-1, :-2] + 1.0,
                padded[1:-1, 2:] + 1.0,
                padded[:-2, 1:-1] + 1.0,
                padded[2:, 1:-1] + 1.0,
            ]),
        )
        if np.array_equal(old, dist):
            break

    max_dist = float(dist[mask_bool].max()) if mask_bool.any() else 1.0
    if max_dist <= 0:
        return np.zeros(mask.shape, dtype=np.float32)
    return (dist / max_dist).clip(0.0, 1.0).astype(np.float32)


def compute_height_gradients(height: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(height.astype(np.float32))
    return np.stack([grad_x, grad_y], axis=0).astype(np.float32)


def compute_laplacian(height: np.ndarray) -> np.ndarray:
    padded = np.pad(height.astype(np.float32), 1, mode="edge")
    laplacian = (
        padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return laplacian.astype(np.float32)


class TerrainRepairDataset(TerrainDiffusionDataset):
    """Assemble deterministic repair samples with realistic masks and terrain features."""

    def __init__(
        self,
        export_dir: str | Path,
        tile_size: int = 128,
        stride_chunks: int = 1,
        mask_mode: str = "mixed",
        mask_fraction_range: tuple[float, float] = (0.15, 0.5),
        seed: int = 0,
        cache_arrays: bool = True,
        height_range: tuple[float, float] | None = None,
        prefill_iterations: int = 64,
    ):
        super().__init__(
            export_dir=export_dir,
            tile_size=tile_size,
            stride_chunks=stride_chunks,
            mask_mode=mask_mode,
            mask_fraction_range=mask_fraction_range,
            seed=seed,
            cache_arrays=cache_arrays,
            height_range=height_range,
        )
        self.prefill_iterations = prefill_iterations

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        origin_x, origin_z = self.window_origins[index]
        surface = self._assemble_surface_window(origin_x, origin_z)
        materials = self._assemble_material_window(origin_x, origin_z)
        support = self._assemble_support_window(origin_x, origin_z)

        target_height = self._normalize_height(surface).astype(np.float32)
        mask = self._build_mask(index)
        known_height = target_height * (1.0 - mask)
        known_material = materials.copy()
        known_material[mask.astype(bool)] = UNKNOWN_INDEX
        known_support = support * (1.0 - mask)
        prefill_height = build_prefill_height(target_height, mask, iterations=self.prefill_iterations)
        boundary_distance = compute_boundary_distance(mask)
        prefill_gradients = compute_height_gradients(prefill_height)
        prefill_laplacian = compute_laplacian(prefill_height)

        sample = TerrainRepairSample(
            target_height=torch.from_numpy(target_height[None, ...]).float(),
            target_material=torch.from_numpy(materials).long(),
            target_support=torch.from_numpy(support[None, ...]).float(),
            known_height=torch.from_numpy(known_height[None, ...]).float(),
            known_material=torch.from_numpy(known_material).long(),
            known_support=torch.from_numpy(known_support[None, ...]).float(),
            mask=torch.from_numpy(mask[None, ...]).float(),
            prefill_height=torch.from_numpy(prefill_height[None, ...]).float(),
            boundary_distance=torch.from_numpy(boundary_distance[None, ...]).float(),
            prefill_gradients=torch.from_numpy(prefill_gradients).float(),
            prefill_laplacian=torch.from_numpy(prefill_laplacian[None, ...]).float(),
            origin_chunk_x=origin_x,
            origin_chunk_z=origin_z,
        )
        return {
            "target_height": sample.target_height,
            "target_material": sample.target_material,
            "target_support": sample.target_support,
            "known_height": sample.known_height,
            "known_material": sample.known_material,
            "known_support": sample.known_support,
            "mask": sample.mask,
            "prefill_height": sample.prefill_height,
            "boundary_distance": sample.boundary_distance,
            "prefill_gradients": sample.prefill_gradients,
            "prefill_laplacian": sample.prefill_laplacian,
            "origin_chunk_x": torch.tensor(sample.origin_chunk_x),
            "origin_chunk_z": torch.tensor(sample.origin_chunk_z),
        }

    def _build_mask(self, index: int) -> np.ndarray:
        if self.mask_mode == "none":
            return np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        if self.mask_mode == "rectangle":
            return self._build_rectangle_mask(index)
        if self.mask_mode == "strip":
            return self._build_strip_mask(index)
        if self.mask_mode == "blob":
            return self._build_blob_mask(index)
        if self.mask_mode == "mixed":
            rng = np.random.default_rng(self.seed + index)
            choice = rng.choice(["rectangle", "strip", "blob", "blob"], p=[0.3, 0.2, 0.25, 0.25])
            if choice == "rectangle":
                return self._build_rectangle_mask(index)
            if choice == "strip":
                return self._build_strip_mask(index)
            return self._build_blob_mask(index)
        raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")

    def _build_rectangle_mask(self, index: int) -> np.ndarray:
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

    def _build_strip_mask(self, index: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + index)
        min_frac, max_frac = self.mask_fraction_range
        max_width = max(3, int(self.tile_size * min(max_frac, 0.25)))
        min_width = max(2, int(self.tile_size * min(min_frac, 0.12)))
        strip_width = int(rng.integers(min_width, max_width + 1))
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        if bool(rng.integers(0, 2)):
            top = int(rng.integers(0, self.tile_size - strip_width + 1))
            mask[top:top + strip_width, :] = 1.0
        else:
            left = int(rng.integers(0, self.tile_size - strip_width + 1))
            mask[:, left:left + strip_width] = 1.0
        return mask

    def _build_blob_mask(self, index: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + index)
        yy, xx = np.mgrid[0:self.tile_size, 0:self.tile_size]
        mask = np.zeros((self.tile_size, self.tile_size), dtype=bool)
        blob_count = int(rng.integers(2, 7))
        min_frac, max_frac = self.mask_fraction_range
        min_radius = max(3, int(self.tile_size * min_frac * 0.35))
        max_radius = max(min_radius + 1, int(self.tile_size * max_frac * 0.45))
        for _ in range(blob_count):
            cy = float(rng.integers(0, self.tile_size))
            cx = float(rng.integers(0, self.tile_size))
            ry = float(rng.integers(min_radius, max_radius + 1))
            rx = float(rng.integers(min_radius, max_radius + 1))
            mask |= (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
        if not mask.any():
            mask[self.tile_size // 2, self.tile_size // 2] = True
        return mask.astype(np.float32)

    def _assemble_support_window(self, origin_x: int, origin_z: int) -> np.ndarray:
        window = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        for dx in range(self.chunks_per_side):
            for dz in range(self.chunks_per_side):
                coord = (origin_x + dx, origin_z + dz)
                tile = compute_support_from_chunk(self._load_chunk(coord)).T
                row = dz * CHUNK_SIZE
                col = dx * CHUNK_SIZE
                window[row:row + CHUNK_SIZE, col:col + CHUNK_SIZE] = tile
        return window


__all__ = [
    "TerrainRepairDataset",
    "TerrainRepairSample",
    "build_prefill_height",
    "compute_boundary_distance",
    "compute_height_gradients",
    "compute_laplacian",
    "compute_support_from_chunk",
    "estimate_support_from_material",
    "solid_mask",
]
