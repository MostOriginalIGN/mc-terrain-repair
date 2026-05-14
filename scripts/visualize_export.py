"""CLI entrypoint for rendering validation images from exported terrain arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPORTER_SRC = ROOT / "packages" / "exporter" / "src"

src_path = str(EXPORTER_SRC)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from exporter.visualize import (  # noqa: E402
    render_colormap,
    render_cross_section,
    render_export_colormap,
    render_export_heightmap,
    render_heightmap,
)

SURFACE_FILENAME_RE = re.compile(r"^surface_(-?\d+)_(-?\d+)\.npy$")


def _discover_surface_files(export_dir: Path) -> list[tuple[int, int, Path]]:
    surface_files: list[tuple[int, int, Path]] = []
    for path in sorted(export_dir.glob("surface_*.npy")):
        match = SURFACE_FILENAME_RE.match(path.name)
        if match is None:
            continue
        surface_files.append((int(match.group(1)), int(match.group(2)), path))
    return surface_files


def _chunk_path_for_surface(surface_path: Path, chunk_x: int, chunk_z: int) -> Path:
    chunk_path = surface_path.with_name(f"chunk_{chunk_x}_{chunk_z}.npy")
    if not chunk_path.exists():
        raise FileNotFoundError(
            f"Missing chunk file for surface_{chunk_x}_{chunk_z}.npy: expected {chunk_path.name}"
        )
    return chunk_path


def _render_sample_previews(
    export_dir: Path,
    out_dir: Path,
    sample_count: int,
    z_slice: int,
    seed: int | None,
) -> int:
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    surface_files = _discover_surface_files(export_dir)
    if not surface_files:
        raise FileNotFoundError(f"No surface_*.npy files found in {export_dir}")

    selected = list(surface_files)
    random.Random(seed).shuffle(selected)
    selected = selected[: min(sample_count, len(selected))]

    for chunk_x, chunk_z, surface_path in selected:
        chunk_path = _chunk_path_for_surface(surface_path, chunk_x, chunk_z)
        surface_y = np.load(surface_path)
        blocks = np.load(chunk_path)

        render_heightmap(surface_y, str(sample_dir / f"heightmap_{chunk_x}_{chunk_z}.png"))
        render_colormap(blocks, str(sample_dir / f"colormap_{chunk_x}_{chunk_z}.png"))
        render_cross_section(
            surface_y,
            blocks,
            z_slice,
            str(sample_dir / f"cross_section_{chunk_x}_{chunk_z}.png"),
        )

    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render validation images from exported terrain arrays.")
    parser.add_argument("--export-dir", required=True, help="Directory containing exported chunk and surface `.npy` files")
    parser.add_argument("--out-dir", default=None, help="Directory for rendered outputs; defaults to <export-dir>/renders")
    parser.add_argument(
        "--heightmap-out",
        default=None,
        help="Optional explicit PNG path for the stitched height map; defaults to <out-dir>/overview_heightmap.png",
    )
    parser.add_argument(
        "--colormap-out",
        default=None,
        help="Optional explicit PNG path for the stitched color map; defaults to <out-dir>/overview_colormap.png",
    )
    parser.add_argument("--sample-count", type=int, default=5, help="How many chunk previews to render")
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic preview sampling")
    parser.add_argument("--z-slice", type=int, default=8, help="Z slice index for cross-section renders")
    args = parser.parse_args()

    export_dir = Path(args.export_dir).expanduser().resolve()
    if not export_dir.is_dir():
        raise SystemExit(f"Export directory does not exist: {export_dir}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else (export_dir / "renders").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    heightmap_out = (
        Path(args.heightmap_out).expanduser().resolve()
        if args.heightmap_out is not None
        else out_dir / "overview_heightmap.png"
    )
    colormap_out = (
        Path(args.colormap_out).expanduser().resolve()
        if args.colormap_out is not None
        else out_dir / "overview_colormap.png"
    )

    rendered, image_size = render_export_heightmap(export_dir, heightmap_out)
    rendered_color, color_image_size = render_export_colormap(export_dir, colormap_out)
    preview_count = _render_sample_previews(
        export_dir=export_dir,
        out_dir=out_dir,
        sample_count=args.sample_count,
        z_slice=args.z_slice,
        seed=args.seed,
    )
    print(
        f"Rendered stitched height map from {rendered} chunks to {heightmap_out} "
        f"at {image_size[0]}x{image_size[1]}"
    )
    print(
        f"Rendered stitched color map from {rendered_color} chunks to {colormap_out} "
        f"at {color_image_size[0]}x{color_image_size[1]}"
    )
    print(f"Rendered {preview_count} sample chunk previews to {out_dir / 'samples'}")


if __name__ == "__main__":
    main()
