from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from exporter import reader
from exporter.export import export_chunks
from exporter.reader import ChunkData
from exporter.visualize import render_export_gallery
from exporter.vocab import NUM_CLASSES, canonical_vocab_payload, encode, vocab_config_path


UNKNOWN_INDEX = 16


@dataclass
class FakeBlock:
    namespace: str
    id: str


class FakeChunk:
    def __init__(self) -> None:
        self.x = 10
        self.z = -4
        self.data = {"Status": "minecraft:full"}
        self.loaded_sections: list[int] = []

    def get_block(self, x: int, y: int, z: int) -> FakeBlock | None:
        if y < -64 or y > 319:
            return None
        if y > 90:
            return None
        if y == 90:
            return FakeBlock("minecraft", "oak_log")
        if y == 89:
            return FakeBlock("minecraft", "water")
        if y == 88:
            return FakeBlock("minecraft", "grass_block")
        if y >= 80:
            return FakeBlock("minecraft", "dirt")
        return FakeBlock("minecraft", "stone")

    def get_section(self, section_y: int) -> object | None:
        return object()

    def stream_blocks(self, section: int | None = None, force_new: bool = False):
        assert section is not None
        self.loaded_sections.append(section)
        base_y = section * 16
        for local_y in range(16):
            world_y = base_y + local_y
            for z in range(16):
                for x in range(16):
                    block = self.get_block(x, world_y, z)
                    if block is None:
                        yield FakeBlock("minecraft", "air")
                    else:
                        yield block


class FakeRegion:
    @classmethod
    def from_file(cls, path: str) -> "FakeRegion":
        return cls()


class FakeAnvil:
    Region = FakeRegion

    class Chunk:
        @staticmethod
        def from_region(region: FakeRegion, local_x: int, local_z: int) -> FakeChunk:
            if local_x == 0 and local_z == 0:
                return FakeChunk()

            class ChunkNotFound(Exception):
                pass

            raise ChunkNotFound("missing")


class MixedChunk(FakeChunk):
    def get_block(self, x: int, y: int, z: int) -> FakeBlock | None:
        if x == 0 and z == 0:
            if y > 120:
                return None
            if y >= 112:
                return FakeBlock("minecraft", "water")
            if y == 111:
                return FakeBlock("minecraft", "sand")
            return FakeBlock("minecraft", "stone")
        return super().get_block(x, y, z)


class MixedAnvil(FakeAnvil):
    class Chunk:
        @staticmethod
        def from_region(region: FakeRegion, local_x: int, local_z: int) -> MixedChunk:
            if local_x == 0 and local_z == 0:
                return MixedChunk()

            class ChunkNotFound(Exception):
                pass

            raise ChunkNotFound("missing")


def _chunk(chunk_x: int, chunk_z: int, fill: int = 1) -> ChunkData:
    return ChunkData(
        chunk_x=chunk_x,
        chunk_z=chunk_z,
        surface_y=np.full((16, 16), 90, dtype=np.int16),
        blocks=np.full((16, 16, 40), fill, dtype=np.int8),
    )


def test_export_pipeline_reader_vocab_and_visualization(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(reader, "anvil", FakeAnvil)

    chunk_ref = reader.ChunkRef(
        region_path=reader.Path("r.0.0.mca"),
        chunk_x=0,
        chunk_z=0,
        local_x=0,
        local_z=0,
    )
    stats = reader.ReaderStats()
    chunk_data = reader.read_chunk(chunk_ref, stats)

    assert chunk_data is not None
    assert chunk_data.surface_y.shape == (16, 16)
    assert chunk_data.blocks.shape == (16, 16, 40)
    assert np.all(chunk_data.surface_y == 88)
    assert chunk_data.blocks.dtype == np.int8
    assert stats.skipped_errors == 0
    assert stats.skipped_not_full == 0

    assert encode("minecraft:grass_block") == 1
    assert encode("minecraft:mycelium") == 15
    assert encode("minecraft:oak_planks") == 0
    assert NUM_CLASSES == 17
    payload = json.loads(vocab_config_path().read_text(encoding="utf-8"))
    assert payload == canonical_vocab_payload()

    chunks_by_coords = {
        (1, 2): _chunk(1, 2),
        (3, 4): _chunk(3, 4, fill=UNKNOWN_INDEX),
    }

    monkeypatch.setattr(
        "exporter.export.iter_chunk_coordinates",
        lambda world_path: [
            ("/tmp/r.0.0.mca", 1, 2, 0, 0),
            ("/tmp/r.0.0.mca", 3, 4, 0, 0),
            ("/tmp/r.0.0.mca", 9, 9, 0, 0),
        ],
    )
    monkeypatch.setattr("exporter.export.anvil", FakeAnvil)

    def fake_read_chunk(ref, export_stats):
        if (ref.chunk_x, ref.chunk_z) == (9, 9):
            export_stats.skipped_not_full += 2
            export_stats.skipped_errors += 1
            return None
        return chunks_by_coords[(ref.chunk_x, ref.chunk_z)]

    monkeypatch.setattr("exporter.export.read_chunk", fake_read_chunk)

    out_dir = tmp_path / "chunks"
    export_chunks("/tmp/world", str(out_dir))

    assert (out_dir / "chunk_1_2.npy").exists()
    assert (out_dir / "surface_1_2.npy").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunk_count"] == 2
    assert manifest["unknown_block_count"] == 16 * 16 * 40
    assert manifest["skipped_not_full"] == 2
    assert manifest["skipped_errors"] == 1
    assert manifest["world_path"] == str(Path("/tmp/world").resolve())

    render_dir = tmp_path / "renders"
    rendered = render_export_gallery(out_dir, render_dir)
    assert rendered == 2
    assert (render_dir / "overview.png").exists()
    assert (render_dir / "chunks" / "chunk_1_2.png").exists()
    overview = Image.open(render_dir / "overview.png")
    assert overview.width > 0
    assert overview.height > 0


def test_reader_uses_section_cache_and_ignores_water_surface(monkeypatch) -> None:
    monkeypatch.setattr(reader, "anvil", MixedAnvil)

    chunk_ref = reader.ChunkRef(
        region_path=reader.Path("r.0.0.mca"),
        chunk_x=0,
        chunk_z=0,
        local_x=0,
        local_z=0,
    )
    stats = reader.ReaderStats()
    chunk_data = reader.read_chunk(chunk_ref, stats)

    assert chunk_data is not None
    assert chunk_data.surface_y.dtype == np.int16
    assert chunk_data.blocks.dtype == np.int8
    assert chunk_data.surface_y[0, 0] == 111
    assert chunk_data.blocks[0, 0, 32] == encode("minecraft:sand")
    assert np.all(chunk_data.blocks >= 0)
    assert np.all(chunk_data.blocks < NUM_CLASSES)


def test_reader_skips_upper_air_sections(monkeypatch) -> None:
    monkeypatch.setattr(reader, "anvil", FakeAnvil)
    chunk = FakeChunk()
    sampler = reader._BlockSampler(chunk)

    anchor_y = reader._find_surface_y(sampler, 0, 0)

    assert anchor_y == 88
    assert 19 in chunk.loaded_sections
    assert 18 in chunk.loaded_sections
    assert 6 in chunk.loaded_sections
    assert 5 in chunk.loaded_sections
    assert 4 not in chunk.loaded_sections


def test_run_export_resolves_save_root_to_overworld(tmp_path) -> None:
    import importlib.util

    script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_export.py"
    spec = importlib.util.spec_from_file_location("run_export_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    save_root = tmp_path / "My World"
    overworld = save_root / "dimensions" / "minecraft" / "overworld"
    (overworld / "region").mkdir(parents=True)

    assert module.resolve_world_path(str(save_root)) == overworld.resolve()
    assert module.resolve_world_path(str(overworld)) == overworld.resolve()
