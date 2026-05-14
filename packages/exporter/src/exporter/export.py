"""Export pipeline for surface-anchored terrain chunks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import functools
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time

import anvil
import numpy as np
from tqdm.auto import tqdm

from dataset.schema import chunk_blocks_filename, manifest_path, surface_filename

from .reader import ChunkData, ChunkRef, ChunkWorkItem, ReaderStats, iter_chunk_coordinates, read_chunk
from .vocab import UNKNOWN_INDEX, vocab_config_path


@functools.lru_cache(maxsize=8)
def _worker_open_region(path_str: str):
    return anvil.Region.from_file(path_str)


def _worker_read_chunk(item: ChunkWorkItem) -> ChunkData | None:
    path_str, chunk_x, chunk_z, local_x, local_z = item
    region = _worker_open_region(path_str)
    ref = ChunkRef(Path(path_str), chunk_x, chunk_z, local_x, local_z, region=region)
    stats = ReaderStats()
    return read_chunk(ref, stats)


def export_chunks(
    world_path: str,
    output_dir: str,
    limit: int | None = None,
    seed: int | None = None,
    workers: int = 1,
) -> None:
    """Export chunks to ``.npy`` (16×16 surface heights + 16×16×40 blocks). Matches ``TerrainDiffusionDataset`` inputs.

    ``workers``: process count; default ``1``. Use ``os.cpu_count()`` or the CLI default for parallelism.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, workers)

    stats = ReaderStats()
    manifest_stats = ManifestStats()
    coords: list[ChunkWorkItem] = list(iter_chunk_coordinates(world_path))
    total_candidates = len(coords)
    progress = ExportProgress(total_candidates=total_candidates, limit=limit)

    if seed is not None:
        random.Random(seed).shuffle(coords)

    if n_workers == 1:
        _export_single(coords, out_dir, stats, manifest_stats, progress, limit)
    else:
        _export_parallel(coords, out_dir, stats, manifest_stats, progress, limit, n_workers)

    progress.close(stats, manifest_stats)
    _write_manifest(
        out_dir=out_dir,
        world_path=world_path,
        manifest_stats=manifest_stats,
        stats=stats,
    )


def _export_single(
    coords: list[ChunkWorkItem],
    out_dir: Path,
    stats: ReaderStats,
    manifest_stats: ManifestStats,
    progress: ExportProgress,
    limit: int | None,
) -> None:
    coords_sorted = sorted(coords, key=lambda c: c[0])
    region = None
    current_path: str | None = None

    for path_str, chunk_x, chunk_z, local_x, local_z in coords_sorted:
        progress.observe_scan(stats, manifest_stats)
        if path_str != current_path:
            region = anvil.Region.from_file(path_str)
            current_path = path_str

        ref = ChunkRef(Path(path_str), chunk_x, chunk_z, local_x, local_z, region=region)
        chunk_data = read_chunk(ref, stats)
        if chunk_data is None:
            progress.refresh(stats, manifest_stats)
            continue

        _save_chunk(out_dir, chunk_data)
        manifest_stats.observe(chunk_data)
        progress.observe_export(stats, manifest_stats)

        if limit is not None and manifest_stats.chunk_count >= limit:
            break


def _export_parallel(
    coords: list[ChunkWorkItem],
    out_dir: Path,
    stats: ReaderStats,
    manifest_stats: ManifestStats,
    progress: ExportProgress,
    limit: int | None,
    n_workers: int,
) -> None:
    with mp.Pool(processes=n_workers) as pool:
        for chunk_data in pool.imap_unordered(_worker_read_chunk, coords, chunksize=4):
            progress.observe_scan(stats, manifest_stats)

            if chunk_data is None:
                progress.refresh(stats, manifest_stats)
                continue

            _save_chunk(out_dir, chunk_data)
            manifest_stats.observe(chunk_data)
            progress.observe_export(stats, manifest_stats)

            if limit is not None and manifest_stats.chunk_count >= limit:
                pool.terminate()
                break


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
        self.start_time = time.perf_counter()
        self._last_postfix = (-1, -1, -1, -1)
        self._export_only = limit is not None
        total = limit if self._export_only else total_candidates
        unit = 'chunk' if self._export_only else 'candidate'
        self._bar = tqdm(total=total, unit=unit, desc='Exporting', dynamic_ncols=True)

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
        elapsed = max(time.perf_counter() - self.start_time, 1e-9)
        exported = manifest_stats.chunk_count
        export_rate = exported / elapsed
        postfix = (exported, stats.skipped_not_full, stats.skipped_errors, self.scanned)
        if postfix == self._last_postfix:
            return
        self._last_postfix = postfix
        payload = {
            'exported': exported,
            'export_chunk/s': f'{export_rate:.2f}',
            'skipped_nf': stats.skipped_not_full,
            'errors': stats.skipped_errors,
        }
        if self._export_only:
            payload['scanned'] = self.scanned
        self._bar.set_postfix(payload)

    def close(self, stats: ReaderStats, manifest_stats: ManifestStats) -> None:
        self.refresh(stats, manifest_stats)
        self._bar.close()


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
