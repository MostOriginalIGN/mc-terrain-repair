"""Export pipeline for surface-anchored terrain chunks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from tqdm.auto import tqdm

from dataset.schema import chunk_blocks_filename, manifest_path, surface_filename

from .reader import ChunkData, ReaderStats, iter_chunk_refs, read_chunk
from .vocab import UNKNOWN_INDEX, vocab_config_path


def export_chunks(
    world_path: str,
    output_dir: str,
    limit: int | None = None,
    seed: int | None = None,
) -> None:
    """Export terrain chunks from a Minecraft world to `.npy` arrays."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = ReaderStats()
    manifest_stats = ManifestStats()
    total_candidates = _count_candidate_chunks(world_path)
    progress = ExportProgress(total_candidates=total_candidates, limit=limit)

    chunk_refs = list(iter_chunk_refs(world_path)) if seed is not None else iter_chunk_refs(world_path)
    if seed is not None:
        random.Random(seed).shuffle(chunk_refs)

    for chunk_ref in chunk_refs:
        progress.observe_scan(stats, manifest_stats)
        chunk_data = read_chunk(chunk_ref, stats)
        if chunk_data is None:
            progress.refresh(stats, manifest_stats)
            continue

        _save_chunk(out_dir, chunk_data)
        manifest_stats.observe(chunk_data)
        progress.observe_export(stats, manifest_stats)

        if limit is not None and manifest_stats.chunk_count >= limit:
            break

    progress.close(stats, manifest_stats)
    _write_manifest(
        out_dir=out_dir,
        world_path=world_path,
        manifest_stats=manifest_stats,
        stats=stats,
    )


@dataclass
class ManifestStats:
    chunk_count: int = 0
    min_chunk_x: int | None = None
    max_chunk_x: int | None = None
    min_chunk_z: int | None = None
    max_chunk_z: int | None = None
    unknown_block_count: int = 0
    total_block_count: int = 0

    def observe(self, chunk_data: ChunkData) -> None:
        self.chunk_count += 1
        self.min_chunk_x = chunk_data.chunk_x if self.min_chunk_x is None else min(self.min_chunk_x, chunk_data.chunk_x)
        self.max_chunk_x = chunk_data.chunk_x if self.max_chunk_x is None else max(self.max_chunk_x, chunk_data.chunk_x)
        self.min_chunk_z = chunk_data.chunk_z if self.min_chunk_z is None else min(self.min_chunk_z, chunk_data.chunk_z)
        self.max_chunk_z = chunk_data.chunk_z if self.max_chunk_z is None else max(self.max_chunk_z, chunk_data.chunk_z)
        self.unknown_block_count += int(np.count_nonzero(chunk_data.blocks == UNKNOWN_INDEX))
        self.total_block_count += int(chunk_data.blocks.size)


class ExportProgress:
    def __init__(self, total_candidates: int, limit: int | None):
        self.total_candidates = total_candidates
        self.limit = limit
        self.scanned = 0
        self._last_postfix = (-1, -1, -1)
        total = limit if limit is not None else total_candidates
        self._export_only = limit is not None
        self._bar = tqdm(total=total, unit='chunk', desc='Exporting', dynamic_ncols=True)

    def observe_scan(self, stats: ReaderStats, manifest_stats: ManifestStats) -> None:
        self.scanned += 1
        if not self._export_only:
            self._bar.update(1)
        self.refresh(stats, manifest_stats)

    def observe_export(self, stats: ReaderStats, manifest_stats: ManifestStats) -> None:
        if self._export_only:
            self._bar.update(1)
        self.refresh(stats, manifest_stats)

    def refresh(self, stats: ReaderStats, manifest_stats: ManifestStats) -> None:
        postfix = (manifest_stats.chunk_count, stats.skipped_not_full, stats.skipped_errors)
        if postfix == self._last_postfix:
            return
        self._last_postfix = postfix
        if self._export_only:
            self._bar.set_postfix({
                'exported': manifest_stats.chunk_count,
                'skipped_nf': stats.skipped_not_full,
                'errors': stats.skipped_errors,
                'scanned': self.scanned,
            })
        else:
            self._bar.set_postfix({
                'exported': manifest_stats.chunk_count,
                'skipped_nf': stats.skipped_not_full,
                'errors': stats.skipped_errors,
            })

    def close(self, stats: ReaderStats, manifest_stats: ManifestStats) -> None:
        self.refresh(stats, manifest_stats)
        self._bar.close()


def _count_candidate_chunks(world_path: str) -> int:
    region_dir = Path(world_path) / "region"
    return len(list(region_dir.glob("*.mca"))) * 32 * 32


def _save_chunk(out_dir: Path, chunk_data: ChunkData) -> None:
    np.save(out_dir / chunk_blocks_filename(chunk_data.chunk_x, chunk_data.chunk_z), chunk_data.blocks)
    np.save(out_dir / surface_filename(chunk_data.chunk_x, chunk_data.chunk_z), chunk_data.surface_y)


def _write_manifest(
    out_dir: Path,
    world_path: str,
    manifest_stats: ManifestStats,
    stats: ReaderStats,
) -> None:
    unknown_block_rate = (
        manifest_stats.unknown_block_count / manifest_stats.total_block_count
        if manifest_stats.total_block_count
        else 0.0
    )
    vocab_bytes = vocab_config_path().read_bytes()
    vocab_hash = hashlib.sha256(vocab_bytes).hexdigest()

    manifest = {
        "chunk_count": manifest_stats.chunk_count,
        "world_path": str(Path(world_path).resolve()),
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "vocab_version_hash": vocab_hash,
        "min_chunk_x": manifest_stats.min_chunk_x,
        "max_chunk_x": manifest_stats.max_chunk_x,
        "min_chunk_z": manifest_stats.min_chunk_z,
        "max_chunk_z": manifest_stats.max_chunk_z,
        "unknown_block_count": manifest_stats.unknown_block_count,
        "total_block_count": manifest_stats.total_block_count,
        "unknown_block_rate": unknown_block_rate,
        "skipped_not_full": stats.skipped_not_full,
        "skipped_errors": stats.skipped_errors,
    }

    manifest_path(out_dir).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
