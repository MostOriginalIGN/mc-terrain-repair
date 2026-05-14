"""Chunk reading helpers built around anvil-parser2."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .vocab import UNKNOWN_INDEX, encode

import anvil

LOGGER = logging.getLogger(__name__)

MIN_WORLD_Y = -64
MAX_WORLD_Y = 319
MIN_SECTION_Y = MIN_WORLD_Y // 16
MAX_SECTION_Y = MAX_WORLD_Y // 16
SURFACE_DEPTH_BELOW = 32
SURFACE_HEIGHT_ABOVE = 8
SURFACE_WINDOW = SURFACE_DEPTH_BELOW + SURFACE_HEIGHT_ABOVE
FULL_STATUSES = {"full", "minecraft:full"}
AIR_INDEX = 0
WATER_INDEX = 11


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
    """Load one chunk into surface height and a (16, 16, 40) anchored block slab."""
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
                    blocks[x, z, index] = block_sampler.sample_encoded_block(x, sample_y, z)

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


def _decode_palette(palette_nbt) -> np.ndarray:
    """Map section palette tags to int8 vocab indices."""
    out = np.empty(len(palette_nbt), dtype=np.int8)
    for i, tag in enumerate(palette_nbt):
        name = tag["Name"].value
        out[i] = encode(name)
    return out


def _decode_block_states(states_nbt, n_palette: int) -> np.ndarray:
    """Unpack BlockStates longs to 4096 palette indices (YZX order)."""
    raw = np.array(
        [s if s >= 0 else s + (1 << 64) for s in states_nbt.value],
        dtype=np.uint64,
    )

    bits = max(int(np.ceil(np.log2(n_palette))) if n_palette > 1 else 1, 4)
    indices_per_long = 64 // bits
    mask = np.uint64((1 << bits) - 1)

    flat = np.zeros(4096, dtype=np.int32)
    for slot in range(indices_per_long):
        shift = np.uint64(slot * bits)
        out_indices = np.arange(slot, min(slot + indices_per_long * len(raw), 4096), indices_per_long)
        if len(out_indices) == 0:
            break
        flat[out_indices] = ((raw[: len(out_indices)] >> shift) & mask).astype(np.int32)

    return flat


class _BlockSampler:
    """Caches encoded section data for faster repeated block lookups."""

    def __init__(self, chunk: object):
        self.chunk = chunk
        self.section_cache: dict[int, np.ndarray] = {}
        self.section_kind_cache: dict[int, str] = {}
        self.section_surface_cache: dict[int, np.ndarray] = {}

    def sample_encoded_block(self, x: int, y: int, z: int) -> int:
        if y > MAX_WORLD_Y or y < MIN_WORLD_Y:
            return AIR_INDEX

        section_y = y >> 4
        local_y = y & 15
        return int(self._load_encoded_section(section_y)[x, z, local_y])

    def section_kind(self, section_y: int) -> str:
        if section_y not in self.section_kind_cache:
            encoded = self._load_encoded_section(section_y)
            self.section_kind_cache[section_y] = _classify_section(encoded)
        return self.section_kind_cache[section_y]

    def top_terrain_offsets(self, section_y: int) -> np.ndarray:
        if section_y not in self.section_surface_cache:
            encoded = self._load_encoded_section(section_y)
            self.section_surface_cache[section_y] = _top_terrain_offsets(encoded)
        return self.section_surface_cache[section_y]

    def _load_encoded_section(self, section_y: int) -> np.ndarray:
        cached = self.section_cache.get(section_y)
        if cached is not None:
            return cached

        encoded = np.zeros((16, 16, 16), dtype=np.int8)

        if hasattr(self.chunk, "get_section"):
            section = self.chunk.get_section(section_y)

            if section is None:
                self.section_cache[section_y] = encoded
                return encoded

            try:
                if "block_states" in section:
                    palette_nbt = section["block_states"]["palette"]
                    states_tag = section["block_states"].get("data")
                else:
                    palette_nbt = section["Palette"]
                    states_tag = section.get("BlockStates")

                palette = _decode_palette(palette_nbt)

                if len(palette) == 1 or states_tag is None:
                    encoded[:] = palette[0]
                    self.section_cache[section_y] = encoded
                    return encoded

                flat = _decode_block_states(states_tag, len(palette))

                block_encoded = palette[flat].reshape(16, 16, 16)
                encoded = block_encoded.transpose(2, 1, 0).astype(np.int8)

                self.section_cache[section_y] = encoded
                return encoded
            except (TypeError, KeyError, AttributeError):
                encoded.fill(0)

        if hasattr(self.chunk, "stream_blocks"):
            for index, block in enumerate(self.chunk.stream_blocks(section=section_y)):
                local_y = index // 256
                z = (index % 256) // 16
                x = index % 16
                encoded[x, z, local_y] = encode(_block_name(block))

            self.section_cache[section_y] = encoded
            return encoded

        self.section_cache[section_y] = encoded
        return encoded


def _find_surface_y(block_sampler: _BlockSampler, x: int, z: int) -> int:
    for section_y in range(MAX_SECTION_Y, MIN_SECTION_Y - 1, -1):
        if block_sampler.section_kind(section_y) != "terrain":
            continue
        top_offsets = block_sampler.top_terrain_offsets(section_y)
        local_y = int(top_offsets[x, z])
        if local_y >= 0:
            return section_y * 16 + local_y

    return MIN_WORLD_Y


def _classify_section(encoded: np.ndarray) -> str:
    if np.all(encoded == AIR_INDEX):
        return "empty"
    if np.all((encoded == AIR_INDEX) | (encoded == WATER_INDEX)):
        return "water_only"
    return "terrain"


def _top_terrain_offsets(encoded: np.ndarray) -> np.ndarray:
    """Highest solid local_y per (x, z), or -1. ``encoded`` is (x, z, y)."""
    terrain_mask = (
        (encoded != AIR_INDEX)
        & (encoded != WATER_INDEX)
        & (encoded != UNKNOWN_INDEX)
    )
    flipped = terrain_mask[:, :, ::-1]
    has_terrain = flipped.any(axis=2)
    top_local_y = np.where(
        has_terrain,
        15 - np.argmax(flipped, axis=2),
        -1,
    )
    return top_local_y.astype(np.int8)


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