"""Dataset and feature utilities for deterministic terrain repair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from exporter.vocab import AIR_INDEX, UNKNOWN_INDEX

from .data import CHUNK_SIZE, SURFACE_INDEX, TerrainDiffusionDataset

WATER_INDEX = 11
MASK_EPOCH_STRIDE = 1_000_003


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
        export_dir: str | Path | Iterable[str | Path],
        tile_size: int = 128,
        stride_chunks: int = 1,
        mask_mode: str = "mixed",
        augment: bool = False,
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
        self.augment = bool(augment)
        self.prefill_iterations = prefill_iterations
        self.mask_epoch = 0

    def set_mask_epoch(self, epoch: int) -> None:
        """Vary synthetic damage across epochs while keeping each epoch reproducible."""
        self.mask_epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        export_id = self.window_export_ids[index]
        origin_x, origin_z = self.window_origins[index]
        surface = self._assemble_surface_window(origin_x, origin_z, export_id=export_id)
        materials = self._assemble_material_window(origin_x, origin_z, export_id=export_id)
        support = self._assemble_support_window(origin_x, origin_z, export_id=export_id)

        target_height = self._normalize_height(surface).astype(np.float32)
        mask = self._build_mask(index, target_height=target_height, materials=materials, support=support)
        if self.augment:
            target_height, materials, support, mask = self._augment_base_arrays(
                index,
                target_height=target_height,
                materials=materials,
                support=support,
                mask=mask,
            )
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
            "height_scale": torch.tensor(float(self.height_max - self.height_min), dtype=torch.float32),
            "origin_chunk_x": torch.tensor(sample.origin_chunk_x),
            "origin_chunk_z": torch.tensor(sample.origin_chunk_z),
        }

    def _augment_base_arrays(
        self,
        index: int,
        target_height: np.ndarray,
        materials: np.ndarray,
        support: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = self._rng(index, salt=17)

        def _flip_h(arr: np.ndarray) -> np.ndarray:
            return np.flip(arr, axis=1)

        def _flip_v(arr: np.ndarray) -> np.ndarray:
            return np.flip(arr, axis=0)

        def _rot90(arr: np.ndarray, k: int) -> np.ndarray:
            return np.rot90(arr, k=k, axes=(0, 1))

        if float(rng.random()) < 0.5:
            target_height = _flip_h(target_height)
            materials = _flip_h(materials)
            support = _flip_h(support)
            mask = _flip_h(mask)
        if float(rng.random()) < 0.5:
            target_height = _flip_v(target_height)
            materials = _flip_v(materials)
            support = _flip_v(support)
            mask = _flip_v(mask)
        k = int(rng.integers(0, 4))
        if k:
            target_height = _rot90(target_height, k)
            materials = _rot90(materials, k)
            support = _rot90(support, k)
            mask = _rot90(mask, k)

        return (
            np.ascontiguousarray(target_height, dtype=np.float32),
            np.ascontiguousarray(materials, dtype=np.int64),
            np.ascontiguousarray(support, dtype=np.float32),
            np.ascontiguousarray(mask, dtype=np.float32),
        )

    def _rng(self, index: int, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + self.mask_epoch * MASK_EPOCH_STRIDE + index * 1009 + salt)

    def _build_mask(
        self,
        index: int,
        target_height: np.ndarray | None = None,
        materials: np.ndarray | None = None,
        support: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.mask_mode == "none":
            return np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        if self.mask_mode == "rectangle":
            return self._build_rectangle_mask(index)
        if self.mask_mode == "strip":
            return self._build_strip_mask(index)
        if self.mask_mode == "blob":
            return self._build_blob_mask(index)
        if self.mask_mode == "mixed":
            rng = self._rng(index)
            choice = rng.choice(["rectangle", "strip", "blob", "blob"], p=[0.3, 0.2, 0.25, 0.25])
            if choice == "rectangle":
                return self._build_rectangle_mask(index)
            if choice == "strip":
                return self._build_strip_mask(index)
            return self._build_blob_mask(index)
        if self.mask_mode == "terrain_mixed":
            return self._build_terrain_mixed_mask(index, target_height=target_height, materials=materials, support=support)
        if self.mask_mode == "selection_mixed":
            return self._build_selection_mixed_mask(index, target_height=target_height, materials=materials, support=support)
        raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")

    def _build_rectangle_mask(
        self,
        index: int,
        rng: np.random.Generator | None = None,
        center: tuple[int, int] | None = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        rng = rng or self._rng(index)
        min_frac, max_frac = self.mask_fraction_range
        min_size = max(4, int(self.tile_size * min_frac * scale))
        max_size = max(min_size, min(self.tile_size, int(self.tile_size * max_frac * scale)))
        height = int(rng.integers(min_size, max_size + 1))
        width = int(rng.integers(min_size, max_size + 1))
        if center is None:
            top = int(rng.integers(0, self.tile_size - height + 1))
            left = int(rng.integers(0, self.tile_size - width + 1))
        else:
            cy, cx = center
            top = int(np.clip(cy - height // 2 + rng.integers(-height // 4, height // 4 + 1), 0, self.tile_size - height))
            left = int(np.clip(cx - width // 2 + rng.integers(-width // 4, width // 4 + 1), 0, self.tile_size - width))
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        mask[top:top + height, left:left + width] = 1.0
        return mask

    def _build_strip_mask(
        self,
        index: int,
        rng: np.random.Generator | None = None,
        center: tuple[int, int] | None = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        rng = rng or self._rng(index)
        min_frac, max_frac = self.mask_fraction_range
        max_width = max(3, int(self.tile_size * min(max_frac * scale, 0.35)))
        min_width = max(2, int(self.tile_size * min(min_frac * scale, 0.18)))
        strip_width = int(rng.integers(min_width, max_width + 1))
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        if bool(rng.integers(0, 2)):
            top = int(rng.integers(0, self.tile_size - strip_width + 1)) if center is None else int(np.clip(center[0] - strip_width // 2, 0, self.tile_size - strip_width))
            mask[top:top + strip_width, :] = 1.0
        else:
            left = int(rng.integers(0, self.tile_size - strip_width + 1)) if center is None else int(np.clip(center[1] - strip_width // 2, 0, self.tile_size - strip_width))
            mask[:, left:left + strip_width] = 1.0
        return mask

    def _build_blob_mask(
        self,
        index: int,
        rng: np.random.Generator | None = None,
        center: tuple[int, int] | None = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        rng = rng or self._rng(index)
        yy, xx = np.mgrid[0:self.tile_size, 0:self.tile_size]
        mask = np.zeros((self.tile_size, self.tile_size), dtype=bool)
        blob_count = int(rng.integers(2, 7))
        min_frac, max_frac = self.mask_fraction_range
        min_radius = max(3, int(self.tile_size * min_frac * 0.35 * scale))
        max_radius = max(min_radius + 1, int(self.tile_size * max_frac * 0.45 * scale))
        for _ in range(blob_count):
            if center is None:
                cy = float(rng.integers(0, self.tile_size))
                cx = float(rng.integers(0, self.tile_size))
            else:
                cy = float(np.clip(center[0] + rng.integers(-max_radius, max_radius + 1), 0, self.tile_size - 1))
                cx = float(np.clip(center[1] + rng.integers(-max_radius, max_radius + 1), 0, self.tile_size - 1))
            ry = float(rng.integers(min_radius, max_radius + 1))
            rx = float(rng.integers(min_radius, max_radius + 1))
            mask |= (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
        if not mask.any():
            mask[self.tile_size // 2, self.tile_size // 2] = True
        return mask.astype(np.float32)

    def _build_chunk_rectangle_mask(
        self,
        rng: np.random.Generator,
        center: tuple[int, int] | None = None,
        max_chunks: int = 4,
    ) -> np.ndarray:
        max_side_chunks = max(1, min(max_chunks, self.chunks_per_side))
        side_choices = np.arange(1, max_side_chunks + 1)
        weights = np.array([0.36, 0.30, 0.22, 0.12][:max_side_chunks], dtype=np.float64)
        if weights.size < side_choices.size:
            weights = np.pad(weights, (0, side_choices.size - weights.size), constant_values=0.06)
        weights /= weights.sum()

        height_chunks = int(rng.choice(side_choices, p=weights))
        width_chunks = int(rng.choice(side_choices, p=weights))
        if float(rng.random()) < 0.20 and max_side_chunks >= 4:
            if bool(rng.integers(0, 2)):
                height_chunks, width_chunks = 2, min(4, max_side_chunks)
            else:
                height_chunks, width_chunks = min(4, max_side_chunks), 2

        height = height_chunks * CHUNK_SIZE
        width = width_chunks * CHUNK_SIZE
        max_top_chunk = self.chunks_per_side - height_chunks
        max_left_chunk = self.chunks_per_side - width_chunks
        if center is None:
            top_chunk = int(rng.integers(0, max_top_chunk + 1))
            left_chunk = int(rng.integers(0, max_left_chunk + 1))
        else:
            cy, cx = center
            top_chunk = int(np.clip(round((cy - height / 2) / CHUNK_SIZE), 0, max_top_chunk))
            left_chunk = int(np.clip(round((cx - width / 2) / CHUNK_SIZE), 0, max_left_chunk))

        top = top_chunk * CHUNK_SIZE
        left = left_chunk * CHUNK_SIZE
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        mask[top:top + height, left:left + width] = 1.0
        return mask

    def _build_compact_blob_mask(
        self,
        rng: np.random.Generator,
        center: tuple[int, int] | None = None,
    ) -> np.ndarray:
        yy, xx = np.mgrid[0:self.tile_size, 0:self.tile_size]
        mask = np.zeros((self.tile_size, self.tile_size), dtype=bool)
        blob_count = int(rng.integers(1, 4))
        base_radius = int(rng.integers(max(8, self.tile_size // 12), max(12, self.tile_size // 4) + 1))
        if center is None:
            cy = int(rng.integers(base_radius, max(base_radius + 1, self.tile_size - base_radius)))
            cx = int(rng.integers(base_radius, max(base_radius + 1, self.tile_size - base_radius)))
        else:
            cy, cx = center
        for _ in range(blob_count):
            local_radius = max(6, int(base_radius * float(rng.uniform(0.55, 1.05))))
            oy = int(rng.integers(-base_radius, base_radius + 1))
            ox = int(rng.integers(-base_radius, base_radius + 1))
            by = float(np.clip(cy + oy, 0, self.tile_size - 1))
            bx = float(np.clip(cx + ox, 0, self.tile_size - 1))
            ry = float(rng.integers(max(5, local_radius // 2), local_radius + 1))
            rx = float(rng.integers(max(5, local_radius // 2), local_radius + 1))
            mask |= (((yy - by) / ry) ** 2 + ((xx - bx) / rx) ** 2) <= 1.0
        return mask.astype(np.float32)

    def _build_small_hole_mask(self, rng: np.random.Generator) -> np.ndarray:
        size_choices = np.array([8, 12, 16, 24], dtype=np.int64)
        valid_sizes = size_choices[size_choices <= self.tile_size]
        size = int(rng.choice(valid_sizes if valid_sizes.size else np.array([max(4, self.tile_size // 8)])))
        height = size
        width = size if float(rng.random()) < 0.65 else int(min(self.tile_size, size * rng.choice([2, 3])))
        top = int(rng.integers(0, self.tile_size - height + 1))
        left = int(rng.integers(0, self.tile_size - width + 1))
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        mask[top:top + height, left:left + width] = 1.0
        return mask

    def _build_border_strip_mask(self, rng: np.random.Generator) -> np.ndarray:
        mask = self._build_strip_mask(0, rng=rng, scale=0.8)
        if float(rng.random()) < 0.35:
            width = int(rng.integers(CHUNK_SIZE, min(self.tile_size, CHUNK_SIZE * 3) + 1))
            if bool(rng.integers(0, 2)):
                mask[:width, :] = 1.0
                mask[:, :width] = 1.0
            else:
                mask[-width:, :] = 1.0
                mask[:, -width:] = 1.0
        return mask.astype(np.float32)

    def _build_selection_mixed_mask(
        self,
        index: int,
        target_height: np.ndarray | None,
        materials: np.ndarray | None,
        support: np.ndarray | None,
    ) -> np.ndarray:
        rng = self._rng(index)
        bucket = float(rng.random())
        if bucket < 0.45:
            return self._build_chunk_rectangle_mask(rng)
        if bucket < 0.65:
            return self._build_compact_blob_mask(rng)
        if bucket < 0.80:
            return self._build_border_strip_mask(rng)
        if bucket < 0.90 and target_height is not None:
            center = self._sample_hard_terrain_center(rng, target_height, materials=materials, support=support)
            if float(rng.random()) < 0.70:
                return self._build_chunk_rectangle_mask(rng, center=center)
            return self._build_compact_blob_mask(rng, center=center)
        if bucket < 0.95:
            return self._build_small_hole_mask(rng)
        return self._build_stress_mask(index, rng=rng)

    def _build_terrain_mixed_mask(
        self,
        index: int,
        target_height: np.ndarray | None,
        materials: np.ndarray | None,
        support: np.ndarray | None,
    ) -> np.ndarray:
        rng = self._rng(index)
        bucket = float(rng.random())
        if bucket < 0.70:
            return self._build_user_like_mask(index, rng=rng, scale=1.0)
        if bucket < 0.90 and target_height is not None:
            center = self._sample_hard_terrain_center(rng, target_height, materials=materials, support=support)
            return self._build_user_like_mask(index, rng=rng, center=center, scale=1.0)
        return self._build_stress_mask(index, rng=rng)

    def _build_user_like_mask(
        self,
        index: int,
        rng: np.random.Generator,
        center: tuple[int, int] | None = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        choice = rng.choice(["rectangle", "strip", "blob", "blob"], p=[0.35, 0.15, 0.25, 0.25])
        if choice == "rectangle":
            return self._build_rectangle_mask(index, rng=rng, center=center, scale=scale)
        if choice == "strip":
            return self._build_strip_mask(index, rng=rng, center=center, scale=scale)
        return self._build_blob_mask(index, rng=rng, center=center, scale=scale)

    def _build_stress_mask(self, index: int, rng: np.random.Generator) -> np.ndarray:
        if float(rng.random()) < 0.45:
            return self._build_rectangle_mask(index, rng=rng, scale=1.35)
        if float(rng.random()) < 0.75:
            mask = self._build_strip_mask(index, rng=rng, scale=1.5)
        else:
            mask = self._build_blob_mask(index, rng=rng, scale=1.3)
        if float(rng.random()) < 0.5:
            mask[0:max(1, self.tile_size // 32), :] = np.maximum(mask[0:max(1, self.tile_size // 32), :], mask.max())
        return mask.astype(np.float32)

    def _sample_hard_terrain_center(
        self,
        rng: np.random.Generator,
        target_height: np.ndarray,
        materials: np.ndarray | None,
        support: np.ndarray | None,
    ) -> tuple[int, int]:
        grad = compute_height_gradients(target_height)
        score = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
        if support is not None:
            support_grad = compute_height_gradients(support.astype(np.float32))
            score += 0.5 * np.sqrt(support_grad[0] ** 2 + support_grad[1] ** 2)
        if materials is not None:
            water = (materials == WATER_INDEX).astype(np.float32)
            water_grad = compute_height_gradients(water)
            score += np.sqrt(water_grad[0] ** 2 + water_grad[1] ** 2)
        score = score.astype(np.float64)
        score -= score.min()
        if float(score.max()) <= 1e-8:
            return int(rng.integers(0, self.tile_size)), int(rng.integers(0, self.tile_size))
        weights = (score + 1e-6).ravel()
        weights /= weights.sum()
        flat_index = int(rng.choice(weights.size, p=weights))
        return divmod(flat_index, self.tile_size)

    def _assemble_support_window(self, origin_x: int, origin_z: int, export_id: int | None = None) -> np.ndarray:
        resolved_export_id = self._resolve_export_id_for_origin(origin_x, origin_z) if export_id is None else export_id
        window = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        for dx in range(self.chunks_per_side):
            for dz in range(self.chunks_per_side):
                coord = (origin_x + dx, origin_z + dz)
                tile = compute_support_from_chunk(self._load_chunk(coord, export_id=resolved_export_id)).T
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
