"""Export pipeline for surface-anchored terrain chunks."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import functools
import hashlib
import itertools
import json
import math
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

REGION_CACHE_CAPACITY = 8


@dataclass(frozen=True)
class WorkerResult:
    chunk_data: ChunkData | None
    skipped_not_full: int = 0
    skipped_errors: int = 0


@functools.lru_cache(maxsize=REGION_CACHE_CAPACITY)
def _worker_open_region(path_str: str):
    return anvil.Region.from_file(path_str)


def _worker_read_chunk(item: ChunkWorkItem) -> WorkerResult:
    path_str, chunk_x, chunk_z, local_x, local_z = item
    region = _worker_open_region(path_str)
    ref = ChunkRef(Path(path_str), chunk_x, chunk_z, local_x, local_z, region=region)
    stats = ReaderStats()
    chunk_data = read_chunk(ref, stats)
    return WorkerResult(
        chunk_data=chunk_data,
        skipped_not_full=stats.skipped_not_full,
        skipped_errors=stats.skipped_errors,
    )


def _bbox_center_chunk_coords(coords: list[ChunkWorkItem]) -> tuple[int, int]:
    if not coords:
        return 0, 0
    xs = [item[1] for item in coords]
    zs = [item[2] for item in coords]
    return (min(xs) + max(xs)) // 2, (min(zs) + max(zs)) // 2


def _chunk_center_sort_key(chunk_x: int, chunk_z: int, center_x: int, center_z: int) -> tuple[int, float, int, int]:
    dx = chunk_x - center_x
    dz = chunk_z - center_z
    ring = max(abs(dx), abs(dz))
    angle = math.atan2(dz, dx)
    return (ring, angle, chunk_x, chunk_z)


def _order_coords_for_export_inplace(
    coords: list[ChunkWorkItem],
    limit: int | None,
    seed: int | None,
) -> None:
    """Keep region batches contiguous while still preferring center-near chunks for limited exports."""
    if len(coords) <= 1:
        return
    if limit is None and seed is None:
        return

    center_x, center_z = _bbox_center_chunk_coords(coords)
    grouped: OrderedDict[str, list[ChunkWorkItem]] = OrderedDict()
    for item in coords:
        grouped.setdefault(item[0], []).append(item)

    def region_key(path_str: str, items: list[ChunkWorkItem]) -> tuple[int, float, int, int, str]:
        xs = [item[1] for item in items]
        zs = [item[2] for item in items]
        region_center_x = (min(xs) + max(xs)) // 2
        region_center_z = (min(zs) + max(zs)) // 2
        ring, angle, _, _ = _chunk_center_sort_key(region_center_x, region_center_z, center_x, center_z)
        return (ring, angle, region_center_x, region_center_z, path_str)

    rng = random.Random(seed) if seed is not None else None
    ordered: list[ChunkWorkItem] = []
    for _, items in sorted(grouped.items(), key=lambda pair: region_key(pair[0], pair[1])):
        items.sort(key=lambda item: _chunk_center_sort_key(item[1], item[2], center_x, center_z))
        if rng is None:
            ordered.extend(items)
            continue
        for _, group in itertools.groupby(
            items,
            key=lambda item: max(abs(item[1] - center_x), abs(item[2] - center_z)),
        ):
            ring_items = list(group)
            rng.shuffle(ring_items)
            ordered.extend(ring_items)
    coords[:] = ordered


def _merge_worker_stats(stats: ReaderStats, result: WorkerResult) -> None:
    stats.skipped_not_full += result.skipped_not_full
    stats.skipped_errors += result.skipped_errors


def export_chunks(
    world_path: str,
    output_dir: str,
    limit: int | None = None,
    seed: int | None = None,
    workers: int = 1,
) -> None:
    """Export chunks to `.npy` (16x16 surface heights + 16x16x40 blocks)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, workers)

    stats = ReaderStats()
    manifest_stats = ManifestStats()
    coords: list[ChunkWorkItem] = list(iter_chunk_coordinates(world_path))
    total_candidates = len(coords)
    progress = ExportProgress(total_candidates=total_candidates, limit=limit)

    _order_coords_for_export_inplace(coords, limit=limit, seed=seed)

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
    open_regions: OrderedDict[str, anvil.Region] = OrderedDict()

    def region_for(path_str: str) -> anvil.Region:
        if path_str in open_regions:
            open_regions.move_to_end(path_str)
            return open_regions[path_str]
        region = anvil.Region.from_file(path_str)
        open_regions[path_str] = region
        while len(open_regions) > REGION_CACHE_CAPACITY:
            open_regions.popitem(last=False)
        return region

    for path_str, chunk_x, chunk_z, local_x, local_z in coords:
        progress.observe_scan(stats, manifest_stats)
        region = region_for(path_str)
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
        for result in pool.imap(_worker_read_chunk, coords, chunksize=4):
            _merge_worker_stats(stats, result)
            progress.observe_scan(stats, manifest_stats)

            chunk_data = result.chunk_data
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
        'chunk_count': manifest_stats.chunk_count,
        'world_path': str(Path(world_path).resolve()),
        'export_timestamp': datetime.now(timezone.utc).isoformat(),
        'vocab_version_hash': vocab_hash,
        'min_chunk_x': manifest_stats.min_chunk_x,
        'max_chunk_x': manifest_stats.max_chunk_x,
        'min_chunk_z': manifest_stats.min_chunk_z,
        'max_chunk_z': manifest_stats.max_chunk_z,
        'unknown_block_count': manifest_stats.unknown_block_count,
        'total_block_count': manifest_stats.total_block_count,
        'unknown_block_rate': unknown_block_rate,
        'skipped_not_full': stats.skipped_not_full,
        'skipped_errors': stats.skipped_errors,
    }

    manifest_path(out_dir).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
