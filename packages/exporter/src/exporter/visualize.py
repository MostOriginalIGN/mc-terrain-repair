"""Validation renders for exported terrain chunks."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

VOCAB_COLORS: dict[int, tuple[int, int, int]] = {
    0: (200, 225, 255),
    1: (95, 159, 53),
    2: (120, 85, 58),
    3: (97, 72, 52),
    4: (120, 120, 120),
    5: (132, 126, 112),
    6: (224, 211, 154),
    7: (198, 123, 67),
    8: (204, 171, 110),
    9: (245, 248, 255),
    10: (147, 198, 255),
    11: (64, 113, 255),
    12: (160, 174, 183),
    13: (97, 76, 51),
    14: (237, 190, 82),
    15: (86, 66, 44),
    16: (255, 0, 255),
}

TERRAIN_GRADIENT_STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (0.0, (44, 92, 52)),
    (0.28, (88, 148, 74)),
    (0.48, (160, 170, 92)),
    (0.68, (161, 132, 92)),
    (0.84, (132, 132, 132)),
    (1.0, (248, 248, 248)),
]

SURFACE_DEPTH_BELOW = 32
SURFACE_HEIGHT_ABOVE = 8
SURFACE_INDEX = SURFACE_DEPTH_BELOW
CHUNK_FILENAME_RE = re.compile(r"^chunk_(-?\d+)_(-?\d+)\.npy$")
SURFACE_FILENAME_RE = re.compile(r"^surface_(-?\d+)_(-?\d+)\.npy$")
PREVIEW_SIZE = (128, 128)
PREVIEW_PADDING = 12
PREVIEW_LABEL_HEIGHT = 18


def _normalized_heightmap(surface_y: np.ndarray) -> np.ndarray:
    if surface_y.ndim != 2:
        raise ValueError(f"Expected a 2D heightmap array, got shape {surface_y.shape}")

    low = float(surface_y.min())
    high = float(surface_y.max())
    if high == low:
        return np.ones(surface_y.shape, dtype=np.float32)
    return ((surface_y.astype(np.float32) - low) / (high - low)).clip(0.0, 1.0)


def _interpolate_color(value: np.ndarray, low: tuple[int, int, int], high: tuple[int, int, int], blend: np.ndarray) -> np.ndarray:
    low_arr = np.array(low, dtype=np.float32)
    high_arr = np.array(high, dtype=np.float32)
    return low_arr + (high_arr - low_arr) * blend[..., None]


def _terrain_colorize(normalized: np.ndarray) -> np.ndarray:
    image = np.zeros(normalized.shape + (3,), dtype=np.float32)
    for index, (start, start_color) in enumerate(TERRAIN_GRADIENT_STOPS[:-1]):
        end, end_color = TERRAIN_GRADIENT_STOPS[index + 1]
        if index == len(TERRAIN_GRADIENT_STOPS) - 2:
            mask = (normalized >= start) & (normalized <= end)
        else:
            mask = (normalized >= start) & (normalized < end)
        if not np.any(mask):
            continue
        span = max(end - start, 1e-6)
        blend = ((normalized[mask] - start) / span).astype(np.float32)
        image[mask] = _interpolate_color(normalized[mask], start_color, end_color, blend)
    return image.clip(0, 255).astype(np.uint8)


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    positions = np.argwhere(mask > 0)
    if positions.size == 0:
        return None
    top = int(positions[:, 0].min())
    left = int(positions[:, 1].min())
    bottom = int(positions[:, 0].max())
    right = int(positions[:, 1].max())
    return top, left, bottom, right


def _draw_mask_box(image: Image.Image, mask: np.ndarray, outline: tuple[int, int, int] = (255, 36, 36), width: int = 2) -> Image.Image:
    bounds = _mask_bounds(mask)
    if bounds is None:
        return image
    top, left, bottom, right = bounds
    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    draw.rectangle((left, top, right, bottom), outline=outline, width=width)
    return boxed


def _heightmap_image(surface_y: np.ndarray, mask: np.ndarray | None = None) -> Image.Image:
    if surface_y.ndim != 2:
        raise ValueError(f"Expected a 2D heightmap array, got shape {surface_y.shape}")
    normalized = _normalized_heightmap(surface_y)
    image = Image.fromarray(_terrain_colorize(normalized), mode="RGB")
    if mask is not None:
        if mask.shape != surface_y.shape:
            raise ValueError(f"Mask shape {mask.shape} must match heightmap shape {surface_y.shape}")
        image = _draw_mask_box(image, mask)
    return image


def _surface_color_image(blocks: np.ndarray) -> Image.Image:
    if blocks.shape != (16, 16, SURFACE_DEPTH_BELOW + SURFACE_HEIGHT_ABOVE):
        raise ValueError(f"Unexpected block slab shape: {blocks.shape}")

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    surface_blocks = blocks[:, :, SURFACE_INDEX]
    for index, color in VOCAB_COLORS.items():
        image[surface_blocks == index] = color
    return Image.fromarray(image, mode="RGB")


def _cross_section_image(surface_y: np.ndarray, blocks: np.ndarray, z_slice: int) -> Image.Image:
    if surface_y.shape != (16, 16):
        raise ValueError(f"Expected surface_y shape (16, 16), got {surface_y.shape}")
    if blocks.shape != (16, 16, SURFACE_DEPTH_BELOW + SURFACE_HEIGHT_ABOVE):
        raise ValueError(f"Unexpected block slab shape: {blocks.shape}")
    if not 0 <= z_slice < 16:
        raise ValueError(f"z_slice must be in [0, 15], got {z_slice}")

    y_min = int(surface_y[:, z_slice].min()) - SURFACE_DEPTH_BELOW
    y_max = int(surface_y[:, z_slice].max()) + SURFACE_HEIGHT_ABOVE - 1
    image = np.zeros((y_max - y_min + 1, 16, 3), dtype=np.uint8)

    for x in range(16):
        anchor_y = int(surface_y[x, z_slice])
        base_y = anchor_y - SURFACE_DEPTH_BELOW
        for index in range(blocks.shape[2]):
            world_y = base_y + index
            row = y_max - world_y
            image[row, x] = VOCAB_COLORS[int(blocks[x, z_slice, index])]

    return Image.fromarray(image, mode="RGB")


def _fit_preview(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        resample=Image.Resampling.NEAREST,
    )
    preview = Image.new("RGB", size, color=(24, 24, 24))
    offset = ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2)
    preview.paste(resized.convert("RGB"), offset)
    return preview


def _load_export_pairs(export_dir: str | Path) -> list[tuple[int, int, Path, Path]]:
    export_path = Path(export_dir)
    chunk_paths: dict[tuple[int, int], Path] = {}
    surface_paths: dict[tuple[int, int], Path] = {}

    for path in export_path.glob("chunk_*.npy"):
        match = CHUNK_FILENAME_RE.match(path.name)
        if match is None:
            continue
        coords = (int(match.group(1)), int(match.group(2)))
        chunk_paths[coords] = path

    for path in export_path.glob("surface_*.npy"):
        match = SURFACE_FILENAME_RE.match(path.name)
        if match is None:
            continue
        coords = (int(match.group(1)), int(match.group(2)))
        surface_paths[coords] = path

    missing_surfaces = sorted(coords for coords in chunk_paths if coords not in surface_paths)
    missing_chunks = sorted(coords for coords in surface_paths if coords not in chunk_paths)
    if missing_surfaces or missing_chunks:
        problems: list[str] = []
        if missing_surfaces:
            problems.append(f"missing surface files for {missing_surfaces}")
        if missing_chunks:
            problems.append(f"missing chunk files for {missing_chunks}")
        raise FileNotFoundError(", ".join(problems))

    return [
        (chunk_x, chunk_z, chunk_paths[(chunk_x, chunk_z)], surface_paths[(chunk_x, chunk_z)])
        for chunk_x, chunk_z in sorted(chunk_paths)
    ]


def _render_chunk_preview(surface_y: np.ndarray, blocks: np.ndarray, chunk_x: int, chunk_z: int, z_slice: int) -> Image.Image:
    heightmap = _fit_preview(_heightmap_image(surface_y), PREVIEW_SIZE)
    colormap = _fit_preview(_surface_color_image(blocks), PREVIEW_SIZE)
    cross_section = _fit_preview(_cross_section_image(surface_y, blocks, z_slice), PREVIEW_SIZE)

    panel_width = PREVIEW_PADDING * 4 + PREVIEW_SIZE[0] * 3
    panel_height = PREVIEW_PADDING * 2 + PREVIEW_SIZE[1] + PREVIEW_LABEL_HEIGHT
    panel = Image.new("RGB", (panel_width, panel_height), color=(245, 243, 238))
    panel.paste(heightmap, (PREVIEW_PADDING, PREVIEW_PADDING))
    panel.paste(colormap, (PREVIEW_PADDING * 2 + PREVIEW_SIZE[0], PREVIEW_PADDING))
    panel.paste(cross_section, (PREVIEW_PADDING * 3 + PREVIEW_SIZE[0] * 2, PREVIEW_PADDING))

    draw = ImageDraw.Draw(panel)
    draw.text(
        (PREVIEW_PADDING, PREVIEW_PADDING + PREVIEW_SIZE[1] + 2),
        f"chunk ({chunk_x}, {chunk_z})",
        fill=(35, 35, 35),
    )
    return panel


def _compose_overview(previews: list[Image.Image]) -> Image.Image:
    if not previews:
        raise ValueError("Expected at least one preview image")

    columns = max(1, math.ceil(math.sqrt(len(previews))))
    rows = math.ceil(len(previews) / columns)
    tile_width, tile_height = previews[0].size
    gutter = 16
    canvas = Image.new(
        "RGB",
        (
            columns * tile_width + gutter * (columns + 1),
            rows * tile_height + gutter * (rows + 1),
        ),
        color=(230, 228, 223),
    )

    for index, preview in enumerate(previews):
        row = index // columns
        column = index % columns
        x = gutter + column * (tile_width + gutter)
        y = gutter + row * (tile_height + gutter)
        canvas.paste(preview, (x, y))

    return canvas


def render_heightmap(surface_y: np.ndarray, out_path: str | Path, mask: np.ndarray | None = None, upscale: int = 16) -> None:
    """Render a terrain-colored surface heightmap from a 2D Y array."""
    image = _heightmap_image(surface_y, mask=mask)
    if upscale > 1:
        image = image.resize(
            (image.width * upscale, image.height * upscale),
            resample=Image.Resampling.NEAREST,
        )
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_file)


def render_colormap(blocks: np.ndarray, out_path: str) -> None:
    """Render a color surface map from a `(16, 16, 40)` block slab."""
    image = _surface_color_image(blocks).resize((256, 256), resample=Image.Resampling.NEAREST)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_file)


def render_cross_section(surface_y: np.ndarray, blocks: np.ndarray, z_slice: int, out_path: str) -> None:
    """Render one Z cross-section from a surface-relative chunk slab."""
    cross_section = _cross_section_image(surface_y, blocks, z_slice)
    out = cross_section.resize((16 * 8, cross_section.height * 8), resample=Image.Resampling.NEAREST)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_file)


def render_export_gallery(export_dir: str | Path, out_dir: str | Path, z_slice: int = 8) -> int:
    """Render per-chunk previews and an overview image for one export directory."""
    previews_dir = Path(out_dir) / "chunks"
    previews_dir.mkdir(parents=True, exist_ok=True)

    previews: list[Image.Image] = []
    export_pairs = _load_export_pairs(export_dir)
    for chunk_x, chunk_z, chunk_path, surface_path in export_pairs:
        surface_y = np.load(surface_path)
        blocks = np.load(chunk_path)
        preview = _render_chunk_preview(surface_y, blocks, chunk_x, chunk_z, z_slice)
        preview.save(previews_dir / f"chunk_{chunk_x}_{chunk_z}.png")
        previews.append(preview)

    overview = _compose_overview(previews)
    overview.save(Path(out_dir) / "overview.png")
    return len(previews)


def render_export_heightmap(export_dir: str | Path, out_path: str | Path) -> tuple[int, tuple[int, int]]:
    """Render one stitched height map for all exported surface arrays."""
    export_path = Path(export_dir)
    surface_paths: dict[tuple[int, int], Path] = {}

    for path in export_path.glob("surface_*.npy"):
        match = SURFACE_FILENAME_RE.match(path.name)
        if match is None:
            continue
        coords = (int(match.group(1)), int(match.group(2)))
        surface_paths[coords] = path

    if not surface_paths:
        raise FileNotFoundError(f"No surface_*.npy files found in {export_path}")

    chunk_xs = sorted(chunk_x for chunk_x, _ in surface_paths)
    chunk_zs = sorted(chunk_z for _, chunk_z in surface_paths)
    min_chunk_x = chunk_xs[0]
    max_chunk_x = chunk_xs[-1]
    min_chunk_z = chunk_zs[0]
    max_chunk_z = chunk_zs[-1]

    full_height = (max_chunk_z - min_chunk_z + 1) * 16
    full_width = (max_chunk_x - min_chunk_x + 1) * 16
    stitched = np.zeros((full_height, full_width), dtype=np.int16)
    filled = np.zeros((full_height, full_width), dtype=bool)

    for (chunk_x, chunk_z), surface_path in surface_paths.items():
        surface_y = np.load(surface_path)
        if surface_y.shape != (16, 16):
            raise ValueError(f"Expected surface_y shape (16, 16), got {surface_y.shape} from {surface_path}")

        x_offset = (chunk_x - min_chunk_x) * 16
        z_offset = (chunk_z - min_chunk_z) * 16
        stitched[z_offset : z_offset + 16, x_offset : x_offset + 16] = surface_y.T
        filled[z_offset : z_offset + 16, x_offset : x_offset + 16] = True

    normalized = _normalized_heightmap(stitched)
    colored = _terrain_colorize(normalized)
    colored[~filled] = (0, 0, 0)

    image = Image.fromarray(colored, mode="RGB")
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_file)

    return len(surface_paths), image.size


def render_export_colormap(export_dir: str | Path, out_path: str | Path) -> tuple[int, tuple[int, int]]:
    """Render one stitched color surface map for all exported chunk slabs."""
    export_pairs = _load_export_pairs(export_dir)
    if not export_pairs:
        raise FileNotFoundError(f"No chunk_*.npy files found in {Path(export_dir)}")

    chunk_xs = sorted(chunk_x for chunk_x, _, _, _ in export_pairs)
    chunk_zs = sorted(chunk_z for _, chunk_z, _, _ in export_pairs)
    min_chunk_x = chunk_xs[0]
    max_chunk_x = chunk_xs[-1]
    min_chunk_z = chunk_zs[0]
    max_chunk_z = chunk_zs[-1]

    full_height = (max_chunk_z - min_chunk_z + 1) * 16
    full_width = (max_chunk_x - min_chunk_x + 1) * 16
    stitched = np.zeros((full_height, full_width, 3), dtype=np.uint8)
    filled = np.zeros((full_height, full_width), dtype=bool)

    for chunk_x, chunk_z, chunk_path, _surface_path in export_pairs:
        blocks = np.load(chunk_path)
        colors = np.array(_surface_color_image(blocks)).transpose(1, 0, 2)
        x_offset = (chunk_x - min_chunk_x) * 16
        z_offset = (chunk_z - min_chunk_z) * 16
        stitched[z_offset : z_offset + 16, x_offset : x_offset + 16] = colors
        filled[z_offset : z_offset + 16, x_offset : x_offset + 16] = True

    stitched[~filled] = (0, 0, 0)
    image = Image.fromarray(stitched, mode="RGB")
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_file)

    return len(export_pairs), image.size


def main() -> None:
    parser = argparse.ArgumentParser(description="Render validation images for exported chunk arrays.")
    parser.add_argument("--surface", required=True, help="Path to a surface_{x}_{z}.npy file")
    parser.add_argument("--chunk", required=True, help="Path to a chunk_{x}_{z}.npy file")
    parser.add_argument("--out-dir", required=True, help="Output directory for PNG files")
    parser.add_argument("--z-slice", type=int, default=8, help="Z slice for the cross-section render")
    args = parser.parse_args()

    surface_y = np.load(args.surface)
    blocks = np.load(args.chunk)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_heightmap(surface_y, out_dir / "heightmap.png")
    render_colormap(blocks, str(out_dir / "colormap.png"))
    render_cross_section(surface_y, blocks, args.z_slice, str(out_dir / "cross_section.png"))


if __name__ == "__main__":
    main()
