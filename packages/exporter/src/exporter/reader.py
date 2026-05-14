"""Chunk reading helpers built around anvil-parser2."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .vocab import encode

import anvil

LOGGER = logging.getLogger(__name__)

MIN_WORLD_Y = -64
MAX_WORLD_Y = 319
SURFACE_DEPTH_BELOW = 32
SURFACE_HEIGHT_ABOVE = 8
SURFACE_WINDOW = SURFACE_DEPTH_BELOW + SURFACE_HEIGHT_ABOVE
FULL_STATUSES = {"full", "minecraft:full"}


@dataclass(frozen=True)
class ChunkData:
    chunk_x: int
    chunk_z: int
    surface_y: np.ndarray
    blocks: np.ndarray


@dataclass(frozen=True)
class ChunkRef:
    region_path: Path
    chunk_x: int
    chunk_z: int
    local_x: int
    local_z: int
    region: Any | None = None


@dataclass
class ReaderStats:
    skipped_not_full: int = 0
    skipped_errors: int = 0


class ChunkIterator:
    """Iterable wrapper that exposes reader stats after iteration."""

    def __init__(self, world_path: str):
        self.world_path = world_path
        self.stats = ReaderStats()

    def __iter__(self) -> Iterator[ChunkData]:
        for chunk_ref in iter_chunk_refs(self.world_path):
            chunk_data = read_chunk(chunk_ref, self.stats)
            if chunk_data is not None:
                yield chunk_data


def iter_chunks(world_path: str) -> ChunkIterator:
    """Iterate over exported chunk tensors in a Minecraft world."""
    return ChunkIterator(world_path)


def iter_chunk_refs(world_path: str) -> Iterator[ChunkRef]:
    """Yield chunk references across all region files in the world."""
    region_dir = Path(world_path) / "region"
    for region_path in sorted(region_dir.glob("*.mca")):
        region = anvil.Region.from_file(str(region_path))
        region_x, region_z = _parse_region_coords(region_path)
        for local_x in range(32):
            for local_z in range(32):
                yield ChunkRef(
                    region_path=region_path,
                    chunk_x=(region_x * 32) + local_x,
                    chunk_z=(region_z * 32) + local_z,
                    local_x=local_x,
                    local_z=local_z,
                    region=region,
                )


def read_chunk(chunk_ref: ChunkRef, stats: ReaderStats | None = None) -> ChunkData | None:
    """Read one chunk reference into a surface-anchored terrain representation."""
    stats = stats or ReaderStats()
    try:
        region = chunk_ref.region if chunk_ref.region is not None else anvil.Region.from_file(str(chunk_ref.region_path))
        chunk = anvil.Chunk.from_region(region, chunk_ref.local_x, chunk_ref.local_z)
    except Exception as exc:
        if exc.__class__.__name__ == "ChunkNotFound":
            return None
        stats.skipped_errors += 1
        LOGGER.warning(
            "Failed to open chunk (%s, %s): %s",
            chunk_ref.chunk_x,
            chunk_ref.chunk_z,
            exc,
        )
        return None

    try:
        status = _chunk_status(chunk)
        if status not in FULL_STATUSES:
            stats.skipped_not_full += 1
            return None

        block_sampler = _BlockSampler(chunk)
        surface_y = np.zeros((16, 16), dtype=np.int16)
        blocks = np.zeros((16, 16, SURFACE_WINDOW), dtype=np.int8)

        for x in range(16):
            for z in range(16):
                anchor_y = _find_surface_y(block_sampler, x, z)
                surface_y[x, z] = anchor_y
                base_y = anchor_y - SURFACE_DEPTH_BELOW
                for index in range(SURFACE_WINDOW):
                    sample_y = base_y + index
                    block_name = block_sampler.sample_block_name(x, sample_y, z)
                    blocks[x, z, index] = encode(block_name)

        return ChunkData(
            chunk_x=chunk_ref.chunk_x,
            chunk_z=chunk_ref.chunk_z,
            surface_y=surface_y,
            blocks=blocks,
        )
    except Exception as exc:
        stats.skipped_errors += 1
        LOGGER.warning(
            "Failed to parse chunk (%s, %s): %s",
            chunk_ref.chunk_x,
            chunk_ref.chunk_z,
            exc,
        )
        return None



def _parse_region_coords(region_path: Path) -> tuple[int, int]:
    stem_parts = region_path.stem.split(".")
    if len(stem_parts) != 3 or stem_parts[0] != "r":
        raise ValueError(f"Unexpected region file name: {region_path.name}")
    return int(stem_parts[1]), int(stem_parts[2])


def _chunk_status(chunk: object) -> str:
    value = _nbt_lookup(getattr(chunk, "data", None), "Status")
    if value is None:
        return "full"
    return str(getattr(value, "value", value)).lower()


def _nbt_lookup(container: object, key: str) -> object | None:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    try:
        return container[key]
    except Exception:
        return getattr(container, key, None)


class _BlockSampler:
    """Caches decoded section data for faster repeated block lookups."""

    def __init__(self, chunk: object):
        self.chunk = chunk
        self.section_cache: dict[int, np.ndarray] = {}

    def sample_block_name(self, x: int, y: int, z: int) -> str:
        if y > MAX_WORLD_Y:
            return "minecraft:air"
        if y < MIN_WORLD_Y:
            return "minecraft:air"

        if hasattr(self.chunk, "get_section") and hasattr(self.chunk, "stream_blocks"):
            section_y = y // 16
            local_y = y % 16
            section = self.section_cache.get(section_y)
            if section is None:
                section = self._load_section(section_y)
                self.section_cache[section_y] = section
            return str(section[x, z, local_y])

        return _sample_block_name_direct(self.chunk, x, y, z)

    def _load_section(self, section_y: int) -> np.ndarray:
        section_blocks = np.empty((16, 16, 16), dtype=object)
        for index, block in enumerate(self.chunk.stream_blocks(section=section_y, force_new=True)):
            local_y = index // 256
            z = (index % 256) // 16
            x = index % 16
            section_blocks[x, z, local_y] = _block_name(block)
        return section_blocks


def _find_surface_y(block_sampler: _BlockSampler, x: int, z: int) -> int:
    for y in range(MAX_WORLD_Y, MIN_WORLD_Y - 1, -1):
        block_name = block_sampler.sample_block_name(x, y, z)
        encoded = encode(block_name)
        if encoded == 0:
            continue
        if encoded == 11:
            continue
        return y
    return MIN_WORLD_Y


def _sample_block_name_direct(chunk: object, x: int, y: int, z: int) -> str:
    if y > MAX_WORLD_Y:
        return "minecraft:air"
    if y < MIN_WORLD_Y:
        return "minecraft:air"

    block = chunk.get_block(x, y, z)
    if block is None:
        return "minecraft:air"

    return _block_name(block)


def _block_name(block: object) -> str:
    if hasattr(block, "name") and callable(block.name):
        name = block.name()
        if isinstance(name, str) and name:
            return name

    namespace = getattr(block, "namespace", "minecraft")
    block_id = getattr(block, "id", "air")
    if isinstance(block_id, str) and ":" in block_id:
        return block_id
    return f"{namespace}:{block_id}"
