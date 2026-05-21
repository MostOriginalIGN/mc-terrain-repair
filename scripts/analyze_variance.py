"""Analyze height and material variance balance in exported repair windows."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_SRC = ROOT / "packages" / "diffusion" / "src"
EXPORTER_SRC = ROOT / "packages" / "exporter" / "src"
DATASET_SRC = ROOT / "packages" / "dataset" / "src"

for src_path in (str(DIFFUSION_SRC), str(EXPORTER_SRC), str(DATASET_SRC)):
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from diffusion.repair_data import compute_height_gradients  # noqa: E402
from diffusion.repair_training import resolve_training_export_dirs  # noqa: E402
from diffusion.data import SURFACE_INDEX, TerrainDiffusionDataset  # noqa: E402
from exporter.vocab import CLASS_NAMES, UNKNOWN_INDEX  # noqa: E402


@dataclass(frozen=True)
class WindowVarianceStats:
    export: str
    origin_chunk_x: int
    origin_chunk_z: int
    height_std: float
    height_range: float
    roughness: float
    material_entropy: float
    material_diversity: int
    dominant_material: str
    dominant_material_fraction: float


@dataclass(frozen=True)
class WorldVarianceStats:
    export: str
    path: str
    chunk_count: int
    surface_cells: int
    height_min: float
    height_max: float
    height_mean: float
    height_std: float
    height_range: float
    roughness: float
    material_entropy: float
    material_diversity: int
    dominant_material: str
    dominant_material_fraction: float
    material_fractions: dict[str, float]


def _quantile_edges(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    low, high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    return float(low), float(high)


def _bucket(value: float, low_edge: float, high_edge: float) -> str:
    if value <= low_edge:
        return "low"
    if value <= high_edge:
        return "mid"
    return "high"


def _material_stats_from_counts(counts: np.ndarray) -> tuple[float, int, int, float]:
    total = float(counts.sum())
    if total <= 0:
        return 0.0, 0, UNKNOWN_INDEX, 0.0
    probabilities = counts[counts > 0] / total
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    diversity = int(np.count_nonzero(counts))
    dominant_index = int(np.argmax(counts))
    dominant_fraction = float(counts[dominant_index] / total)
    return entropy, diversity, dominant_index, dominant_fraction


def _material_entropy(materials: np.ndarray) -> tuple[float, int, int, float]:
    valid = materials[(materials >= 0) & (materials < UNKNOWN_INDEX)]
    if valid.size == 0:
        return 0.0, 0, UNKNOWN_INDEX, 0.0
    counts = np.bincount(valid.ravel(), minlength=UNKNOWN_INDEX).astype(np.float64)
    return _material_stats_from_counts(counts)


def _material_fractions(counts: np.ndarray) -> dict[str, float]:
    total = float(counts.sum())
    if total <= 0:
        return {}
    fractions: dict[str, float] = {}
    for index, count in enumerate(counts):
        if count <= 0:
            continue
        name = CLASS_NAMES[index] if index < len(CLASS_NAMES) else str(index)
        fractions[name] = float(count / total)
    return dict(sorted(fractions.items(), key=lambda item: item[1], reverse=True))


def _world_stats(dataset: TerrainDiffusionDataset, export_id: int) -> WorldVarianceStats:
    surface_paths = dataset.surface_paths_by_export[export_id]
    height_min = float("inf")
    height_max = float("-inf")
    height_sum = 0.0
    height_sumsq = 0.0
    roughness_sum = 0.0
    cell_count = 0
    material_counts = np.zeros(UNKNOWN_INDEX, dtype=np.float64)

    for coord in sorted(surface_paths):
        surface = dataset._load_surface(coord, export_id=export_id).astype(np.float64)
        chunk = dataset._load_chunk(coord, export_id=export_id)
        materials = chunk[:, :, SURFACE_INDEX]
        valid_materials = materials[(materials >= 0) & (materials < UNKNOWN_INDEX)]
        if valid_materials.size:
            material_counts += np.bincount(valid_materials.ravel(), minlength=UNKNOWN_INDEX)

        gradients = compute_height_gradients(surface.astype(np.float32))
        roughness = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)
        height_min = min(height_min, float(surface.min()))
        height_max = max(height_max, float(surface.max()))
        height_sum += float(surface.sum())
        height_sumsq += float((surface * surface).sum())
        roughness_sum += float(roughness.sum())
        cell_count += int(surface.size)

    if cell_count == 0:
        height_min = 0.0
        height_max = 0.0
    height_mean = height_sum / max(cell_count, 1)
    height_variance = max(0.0, height_sumsq / max(cell_count, 1) - height_mean * height_mean)
    entropy, diversity, dominant_index, dominant_fraction = _material_stats_from_counts(material_counts)
    dominant_name = CLASS_NAMES[dominant_index] if dominant_index < len(CLASS_NAMES) else str(dominant_index)

    return WorldVarianceStats(
        export=dataset.export_dirs[export_id].name,
        path=str(dataset.export_dirs[export_id]),
        chunk_count=len(surface_paths),
        surface_cells=cell_count,
        height_min=height_min,
        height_max=height_max,
        height_mean=height_mean,
        height_std=float(np.sqrt(height_variance)),
        height_range=height_max - height_min,
        roughness=roughness_sum / max(cell_count, 1),
        material_entropy=entropy,
        material_diversity=diversity,
        dominant_material=dominant_name,
        dominant_material_fraction=dominant_fraction,
        material_fractions=_material_fractions(material_counts),
    )


def _window_stats(dataset: TerrainDiffusionDataset, index: int) -> WindowVarianceStats:
    export_id = dataset.window_export_ids[index]
    origin_x, origin_z = dataset.window_origins[index]
    surface = dataset._assemble_surface_window(origin_x, origin_z, export_id=export_id)
    materials = dataset._assemble_material_window(origin_x, origin_z, export_id=export_id)
    gradients = compute_height_gradients(surface.astype(np.float32))
    roughness = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2).mean()
    entropy, diversity, dominant_index, dominant_fraction = _material_entropy(materials)
    dominant_name = CLASS_NAMES[dominant_index] if dominant_index < len(CLASS_NAMES) else str(dominant_index)
    return WindowVarianceStats(
        export=dataset.export_dirs[export_id].name,
        origin_chunk_x=origin_x,
        origin_chunk_z=origin_z,
        height_std=float(surface.std()),
        height_range=float(surface.max() - surface.min()),
        roughness=float(roughness),
        material_entropy=entropy,
        material_diversity=diversity,
        dominant_material=dominant_name,
        dominant_material_fraction=dominant_fraction,
    )


def _choose_indices(total: int, limit: int | None, seed: int) -> list[int]:
    indices = list(range(total))
    if limit is None or limit >= total:
        return indices
    rng = random.Random(seed)
    rng.shuffle(indices)
    return sorted(indices[:limit])


def _histogram(values: Iterable[str]) -> dict[str, int]:
    counts = {"low": 0, "mid": 0, "high": 0}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _recommended_focus(joint_counts: dict[str, int]) -> list[str]:
    if not joint_counts:
        return []
    target = max(1, round(sum(joint_counts.values()) / len(joint_counts)))
    focus = [
        bucket
        for bucket, count in sorted(joint_counts.items(), key=lambda item: (item[1], item[0]))
        if count < target * 0.6
    ]
    return focus[:5]


def analyze_export_variance(
    export_dir: str | Path,
    tile_size: int,
    stride_chunks: int,
    limit_windows: int | None,
    seed: int,
) -> dict[str, object]:
    export_dirs = resolve_training_export_dirs([str(export_dir)])
    dataset = TerrainDiffusionDataset(
        export_dirs,
        tile_size=tile_size,
        stride_chunks=stride_chunks,
        mask_mode="none",
        cache_arrays=True,
    )
    indices = _choose_indices(len(dataset), limit_windows, seed)
    worlds = [_world_stats(dataset, export_id) for export_id in range(len(dataset.export_dirs))]
    windows = [_window_stats(dataset, index) for index in indices]
    height_values = np.array([window.height_std for window in windows], dtype=np.float64)
    material_values = np.array([window.material_entropy for window in windows], dtype=np.float64)
    roughness_values = np.array([window.roughness for window in windows], dtype=np.float64)
    height_low, height_high = _quantile_edges(height_values)
    material_low, material_high = _quantile_edges(material_values)

    window_rows: list[dict[str, object]] = []
    joint_counts: dict[str, int] = {}
    height_buckets: list[str] = []
    material_buckets: list[str] = []
    for window in windows:
        height_bucket = _bucket(window.height_std, height_low, height_high)
        material_bucket = _bucket(window.material_entropy, material_low, material_high)
        joint_bucket = f"height_{height_bucket}_material_{material_bucket}"
        height_buckets.append(height_bucket)
        material_buckets.append(material_bucket)
        joint_counts[joint_bucket] = joint_counts.get(joint_bucket, 0) + 1
        row = asdict(window)
        row["height_bucket"] = height_bucket
        row["material_bucket"] = material_bucket
        row["joint_bucket"] = joint_bucket
        window_rows.append(row)

    return {
        "export_dirs": [str(path) for path in export_dirs],
        "world_count": len(worlds),
        "tile_size": tile_size,
        "stride_chunks": stride_chunks,
        "total_windows": len(dataset),
        "analyzed_windows": len(windows),
        "worlds": [asdict(world) for world in worlds],
        "height_std_edges": {"low_mid": height_low, "mid_high": height_high},
        "material_entropy_edges": {"low_mid": material_low, "mid_high": material_high},
        "height_std": {
            "min": float(height_values.min()) if height_values.size else 0.0,
            "mean": float(height_values.mean()) if height_values.size else 0.0,
            "max": float(height_values.max()) if height_values.size else 0.0,
        },
        "material_entropy": {
            "min": float(material_values.min()) if material_values.size else 0.0,
            "mean": float(material_values.mean()) if material_values.size else 0.0,
            "max": float(material_values.max()) if material_values.size else 0.0,
        },
        "roughness": {
            "min": float(roughness_values.min()) if roughness_values.size else 0.0,
            "mean": float(roughness_values.mean()) if roughness_values.size else 0.0,
            "max": float(roughness_values.max()) if roughness_values.size else 0.0,
        },
        "height_bucket_counts": _histogram(height_buckets),
        "material_bucket_counts": _histogram(material_buckets),
        "joint_bucket_counts": dict(sorted(joint_counts.items())),
        "recommended_focus": _recommended_focus(joint_counts),
        "windows": window_rows,
    }


def _write_csv(path: Path, windows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not windows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)


def _print_summary(payload: dict[str, object], json_path: Path, csv_path: Path | None) -> None:
    print(f"Analyzed {payload['world_count']} world/export summary rows")
    worlds = payload["worlds"]
    if isinstance(worlds, list):
        for world in worlds:
            if not isinstance(world, dict):
                continue
            print(
                f"[{world['export']}] height range={world['height_min']:.1f}..{world['height_max']:.1f} "
                f"std={world['height_std']:.2f} roughness={world['roughness']:.2f}; "
                f"materials={world['material_diversity']} entropy={world['material_entropy']:.2f} "
                f"dominant={world['dominant_material']} ({world['dominant_material_fraction']:.1%})"
            )
    print(f"Analyzed {payload['analyzed_windows']} of {payload['total_windows']} windows")
    print(f"Height std buckets: {payload['height_bucket_counts']}")
    print(f"Material entropy buckets: {payload['material_bucket_counts']}")
    print(f"Joint buckets: {payload['joint_bucket_counts']}")
    focus = payload["recommended_focus"]
    print(f"Recommended focus: {focus if focus else 'already fairly balanced by quantile buckets'}")
    print(f"Wrote JSON report to {json_path}")
    if csv_path is not None:
        print(f"Wrote CSV window table to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze height/material variance balance for repair training windows.")
    parser.add_argument("--export-dir", required=True, help="Export directory or parent directory of exports")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--stride-chunks", type=int, default=1)
    parser.add_argument("--limit-windows", type=int, default=None, help="Optional random window sample limit")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="artifacts/variance_analysis.json")
    parser.add_argument("--csv-out", default="artifacts/variance_windows.csv")
    args = parser.parse_args()

    payload = analyze_export_variance(
        export_dir=args.export_dir,
        tile_size=args.tile_size,
        stride_chunks=args.stride_chunks,
        limit_windows=args.limit_windows,
        seed=args.seed,
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = Path(args.csv_out).expanduser().resolve() if args.csv_out else None
    if csv_path is not None:
        windows = payload["windows"]
        if not isinstance(windows, list):
            raise SystemExit("Internal error: expected windows to be a list")
        _write_csv(csv_path, windows)

    _print_summary(payload, out_path, csv_path)


if __name__ == "__main__":
    main()
