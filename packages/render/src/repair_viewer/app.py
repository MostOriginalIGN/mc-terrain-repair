"""3D block viewer for repair outputs"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEA_LEVEL_WORLD = 64.0

_VOCAB_RGB: list[tuple[int, int, int]] = [
    (200, 225, 255),
    (95, 159, 53),
    (120, 85, 58),
    (97, 72, 52),
    (120, 120, 120),
    (132, 126, 112),
    (224, 211, 154),
    (198, 123, 67),
    (204, 171, 110),
    (245, 248, 255),
    (147, 198, 255),
    (64, 113, 255),
    (160, 174, 183),
    (97, 76, 51),
    (237, 190, 82),
    (86, 66, 44),
    (255, 0, 255),
]

_SHADE = {"top": 1.0, "px": 0.75, "nx": 0.65, "pz": 0.85, "nz": 0.55}


@dataclass
class Scene:
    height: np.ndarray
    material: np.ndarray
    mask: np.ndarray
    sea_y: float
    visible: np.ndarray | None = None


def _discover_cases(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "height.npy").is_file()]


def _try_load_metadata(case_dir: Path, repair_cases_dir: Path) -> tuple[float, float] | None:
    for path in (case_dir / "metadata.json", repair_cases_dir / case_dir.name / "metadata.json"):
        if not path.is_file():
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        hmin, hmax = meta.get("height_min"), meta.get("height_max")
        if isinstance(hmin, (int, float)) and isinstance(hmax, (int, float)):
            return float(hmin), float(hmax)
    return None


def _load_mask(case_dir: Path, repair_cases_dir: Path, ref: np.ndarray) -> np.ndarray:
    for path in (case_dir / "mask.npy", repair_cases_dir / case_dir.name / "mask.npy"):
        if path.is_file():
            return np.load(path).astype(np.float32)
    return np.zeros(ref.shape, dtype=np.float32)


def load_scene(case_dir: Path, repair_cases_dir: Path) -> Scene:
    hw = case_dir / "height_world.npy"
    hn = case_dir / "height.npy"
    if hw.is_file():
        height = np.load(hw).astype(np.float64)
        sea_y = SEA_LEVEL_WORLD
    elif hn.is_file():
        raw = np.load(hn).astype(np.float64)
        meta = _try_load_metadata(case_dir, repair_cases_dir)
        if meta is not None:
            hmin, hmax = meta
            height = raw * (hmax - hmin) + hmin
        else:
            height = raw.astype(np.float64)
        sea_y = SEA_LEVEL_WORLD
    else:
        raise FileNotFoundError(case_dir / "height.npy")

    mat_path = case_dir / "material.npy"
    if not mat_path.is_file():
        raise FileNotFoundError(mat_path)
    material = np.load(mat_path)
    mask = _load_mask(case_dir, repair_cases_dir, height)
    if mask.shape != height.shape:
        raise ValueError(f"mask {mask.shape} != height {height.shape}")
    return Scene(height=np.round(height).astype(np.float64),
                 material=material, mask=mask, sea_y=sea_y)


def _cell_rgb(material: np.ndarray, mask: np.ndarray, r: int, c: int, *, highlight_known: bool = True) -> np.ndarray:
    mi = int(material[r, c]) % len(_VOCAB_RGB)
    base = np.array(_VOCAB_RGB[mi], dtype=np.float32)
    if highlight_known and mask[r, c] <= 0:
        return (base * 0.52 + np.array([70, 72, 78], dtype=np.float32)).clip(0, 255)
    return base


def build_mesh(scene: Scene, *, highlight_known: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H, W = scene.height.shape
    visible = np.ones((H, W), dtype=bool) if scene.visible is None else scene.visible.astype(bool)
    if visible.shape != scene.height.shape:
        raise ValueError(f"visible {visible.shape} != height {scene.height.shape}")
    visible_heights = scene.height[visible]
    if visible_heights.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros(3, dtype=np.float64),
        )
    floor_y = float(visible_heights.min()) - 1.0

    verts_list: list[np.ndarray] = []
    quads_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []
    base_v = 0

    def add_quad(corners: np.ndarray, shade: float, rgb: np.ndarray) -> None:
        nonlocal base_v
        col = np.clip(rgb * shade, 0, 255).astype(np.uint8)
        verts_list.append(corners)
        quads_list.append(np.array([base_v, base_v + 1, base_v + 2, base_v + 3], dtype=np.int32))
        colors_list.append(col)
        base_v += 4

    def neighbor_height(r: int, c: int) -> float:
        if 0 <= r < H and 0 <= c < W and visible[r, c]:
            return float(scene.height[r, c])
        return floor_y

    for r in range(H):
        for c in range(W):
            if not visible[r, c]:
                continue
            y = float(scene.height[r, c])
            x0, x1 = float(c), float(c + 1)
            z0, z1 = float(r), float(r + 1)
            rgb = _cell_rgb(scene.material, scene.mask, r, c, highlight_known=highlight_known)

            add_quad(np.array([
                [x0, y, z0], [x1, y, z0],
                [x1, y, z1], [x0, y, z1],
            ]), _SHADE["top"], rgb)

            ny_px = neighbor_height(r, c + 1)
            if ny_px < y:
                add_quad(np.array([
                    [x1, ny_px, z0], [x1, y,    z0],
                    [x1, y,    z1],  [x1, ny_px, z1],
                ]), _SHADE["px"], rgb)

            ny_nx = neighbor_height(r, c - 1)
            if ny_nx < y:
                add_quad(np.array([
                    [x0, y,    z0], [x0, ny_nx, z0],
                    [x0, ny_nx, z1], [x0, y,    z1],
                ]), _SHADE["nx"], rgb)

            ny_pz = neighbor_height(r + 1, c)
            if ny_pz < y:
                add_quad(np.array([
                    [x0, y,    z1], [x1, y,    z1],
                    [x1, ny_pz, z1], [x0, ny_pz, z1],
                ]), _SHADE["pz"], rgb)

            ny_nz = neighbor_height(r - 1, c)
            if ny_nz < y:
                add_quad(np.array([
                    [x0, ny_nz, z0], [x1, ny_nz, z0],
                    [x1, y,    z0],  [x0, y,    z0],
                ]), _SHADE["nz"], rgb)

    WATER_RGB = np.array([62, 118, 188], dtype=np.float32)
    WATER_SHADE = 0.82
    sy = scene.sea_y
    hmap = scene.height

    for r in range(H):
        c = 0
        while c < W:
            if not visible[r, c] or float(hmap[r, c]) >= sy:
                c += 1
                continue
            c0 = c
            while c < W and visible[r, c] and float(hmap[r, c]) < sy:
                c += 1
            x0, x1 = float(c0), float(c)
            z0, z1 = float(r), float(r + 1)
            add_quad(
                np.array([[x0, sy, z0], [x1, sy, z0], [x1, sy, z1], [x0, sy, z1]]),
                WATER_SHADE,
                WATER_RGB,
            )

    verts = np.concatenate(verts_list, axis=0).astype(np.float64)
    faces = np.stack(quads_list, axis=0)
    colors = np.array(colors_list, dtype=np.uint8)
    center = verts.mean(axis=0)
    return verts - center, faces, colors, center


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pos = np.argwhere(mask > 0)
    if pos.size == 0:
        return None
    return int(pos[:, 0].min()), int(pos[:, 1].min()), int(pos[:, 0].max()), int(pos[:, 1].max())


def _box_corners(top: int, left: int, bottom: int, right: int,
                 y0: float, y1: float, center: np.ndarray) -> np.ndarray:
    x0, x1 = float(left), float(right + 1)
    z0, z1 = float(top),  float(bottom + 1)
    c = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1],
        [x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1],
    ], dtype=np.float64)
    return c - center


def _rot(ax: float, ay: float) -> np.ndarray:
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    return ry @ rx


def _figure_rot(ax: float = -0.70, ay: float = -0.785) -> np.ndarray:
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    return rx @ ry


BOX_EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def _metadata_range(case_dir: Path) -> tuple[float, float] | None:
    path = case_dir / "metadata.json"
    if not path.is_file():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    hmin, hmax = meta.get("height_min"), meta.get("height_max")
    if isinstance(hmin, (int, float)) and isinstance(hmax, (int, float)):
        return float(hmin), float(hmax)
    return None


def _load_height_for_figure(path: Path, reference_dir: Path) -> np.ndarray:
    height = np.load(path).astype(np.float64)
    height_range = _metadata_range(reference_dir)
    if height_range is not None and float(np.nanmax(height)) <= 1.5 and float(np.nanmin(height)) >= -0.1:
        hmin, hmax = height_range
        height = height * (hmax - hmin) + hmin
    return np.round(height).astype(np.float64)


def _figure_scene_from_arrays(
    height: np.ndarray,
    material: np.ndarray,
    *,
    visible: np.ndarray | None = None,
) -> Scene:
    if material.shape != height.shape:
        raise ValueError(f"material {material.shape} != height {height.shape}")
    if visible is not None and visible.shape != height.shape:
        raise ValueError(f"visible {visible.shape} != height {height.shape}")
    return Scene(
        height=height,
        material=material,
        mask=np.ones(height.shape, dtype=np.float32),
        sea_y=SEA_LEVEL_WORLD,
        visible=visible,
    )


def _load_figure_scenes(case_dir: Path, repair_cases_dir: Path) -> list[tuple[str, Scene]]:
    reference_dir = repair_cases_dir / case_dir.name
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Missing reference repair case: {reference_dir}")

    mask_path = case_dir / "mask.npy"
    if not mask_path.is_file():
        mask_path = reference_dir / "mask.npy"
    mask = np.load(mask_path).astype(np.float32)

    known_height = _load_height_for_figure(reference_dir / "known_height.npy", reference_dir)
    known_material = np.load(reference_dir / "known_material.npy")
    target_height = _load_height_for_figure(reference_dir / "target_height.npy", reference_dir)
    target_material = np.load(reference_dir / "target_material.npy")

    repaired_height_path = case_dir / "height_world.npy"
    if not repaired_height_path.is_file():
        repaired_height_path = case_dir / "height.npy"
    repaired_height = _load_height_for_figure(repaired_height_path, reference_dir)
    repaired_material = np.load(case_dir / "material.npy")

    visible_known = mask <= 0
    return [
        ("(a) Masked Input", _figure_scene_from_arrays(known_height, known_material, visible=visible_known)),
        ("(b) Repaired", _figure_scene_from_arrays(repaired_height, repaired_material)),
        ("(c) True Terrain", _figure_scene_from_arrays(target_height, target_material)),
    ]


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_isometric_scene(
    scene: Scene,
    *,
    size: tuple[int, int] = (520, 380),
    margin: int = 18,
    supersample: int = 2,
) -> Image.Image:
    verts, faces, colors, _center = build_mesh(scene, highlight_known=False)
    canvas = Image.new("RGB", size, (255, 255, 255))
    if verts.size == 0 or faces.size == 0:
        return canvas

    aa = max(1, int(supersample))
    work_size = (size[0] * aa, size[1] * aa)
    image = Image.new("RGB", work_size, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    rotation = _figure_rot()
    projected = verts @ rotation.T
    xy = np.column_stack((projected[:, 0], -projected[:, 1]))
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1e-6)
    draw_margin = margin * aa
    scale = min((work_size[0] - 2 * draw_margin) / span[0], (work_size[1] - 2 * draw_margin) / span[1])
    fitted = (xy - xy_min) * scale
    offset = np.array(
        [
            (work_size[0] - span[0] * scale) * 0.5,
            (work_size[1] - span[1] * scale) * 0.52,
        ],
        dtype=np.float64,
    )
    points = fitted + offset

    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    shadow_h = max(8 * aa, int((bounds_max[1] - bounds_min[1]) * 0.08))
    shadow_box = (
        int(bounds_min[0] + 18 * aa),
        int(bounds_max[1] - shadow_h),
        int(bounds_max[0] - 18 * aa),
        int(bounds_max[1] + shadow_h * 0.45),
    )
    draw.ellipse(shadow_box, fill=(236, 238, 240))

    depth = projected[:, 2][faces].mean(axis=1)
    for face_index in np.argsort(-depth):
        indexes = faces[face_index]
        polygon = [(float(points[i, 0]), float(points[i, 1])) for i in indexes]
        col = colors[face_index]
        outline = tuple(int(max(0, min(255, channel * 0.78))) for channel in col)
        fill = (int(col[0]), int(col[1]), int(col[2]))
        draw.polygon(polygon, fill=fill, outline=outline)

    if aa > 1:
        image = image.resize(size, resample=Image.Resampling.LANCZOS)
    return image


def compose_case_figure(
    case_dir: Path,
    repair_cases_dir: Path,
    out_path: Path,
    *,
    panel_size: tuple[int, int] = (520, 380),
) -> Path:
    panels = _load_figure_scenes(case_dir, repair_cases_dir)
    rendered = [(label, render_isometric_scene(scene, size=panel_size)) for label, scene in panels]
    gap = 14
    label_h = 42
    margin = 10
    width = margin * 2 + len(rendered) * panel_size[0] + (len(rendered) - 1) * gap
    height = margin + panel_size[1] + label_h + margin
    figure = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(figure)
    label_font = _font(28)

    for index, (label, image) in enumerate(rendered):
        x = margin + index * (panel_size[0] + gap)
        figure.paste(image, (x, margin))
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_w = bbox[2] - bbox[0]
        draw.text(
            (x + (panel_size[0] - text_w) * 0.5, margin + panel_size[1] + 4),
            label,
            fill=(15, 15, 15),
            font=label_font,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(out_path)
    return out_path


def _compose_figure_contact_sheet(case_paths: list[Path], out_path: Path) -> Path | None:
    if not case_paths:
        return None
    loaded = [Image.open(path).convert("RGB") for path in case_paths if path.is_file()]
    if not loaded:
        return None
    max_width = 1200
    rows: list[Image.Image] = []
    for image in loaded:
        if image.width > max_width:
            scale = max_width / image.width
            image = image.resize((max_width, max(1, int(image.height * scale))), resample=Image.Resampling.LANCZOS)
        rows.append(image)
    gap = 24
    margin = 18
    width = max(image.width for image in rows) + margin * 2
    height = margin * 2 + sum(image.height for image in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = margin
    for image in rows:
        x = (width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def export_figures(
    case_dirs: list[Path],
    repair_cases_dir: Path,
    out_dir: Path,
    *,
    panel_size: tuple[int, int] = (520, 380),
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case_dir in case_dirs:
        out_path = out_dir / f"{case_dir.name}.png"
        written.append(compose_case_figure(case_dir, repair_cases_dir, out_path, panel_size=panel_size))
    _compose_figure_contact_sheet(written, out_dir / "all_cases.png")
    return written


def _run(case_dirs: list[Path], repair_cases_dir: Path) -> None:
    import pygame
    from pygame import Rect

    pygame.init()
    sw, sh = 960, 640
    screen = pygame.display.set_mode((sw, sh))
    pygame.display.set_caption("repair 3d — drag orbit, wheel zoom, [/] case")
    font       = pygame.font.SysFont(None, 22)
    font_small = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()

    case_labels = [p.name for p in case_dirs]
    n_cases = len(case_dirs)
    HEADER_H, ROW_H, DD_MAX = 32, 26, 14
    dd_open, dd_scroll = False, 0
    dd_x, dd_y, dd_pad = 8, 8, 10

    def dd_w() -> int:
        return int(np.clip(
            max((font.size(f"  {label}  ")[0] for label in case_labels), default=200) + 36, 200, 360))
    def hr()       -> Rect: return Rect(dd_x, dd_y, dd_w(), HEADER_H)
    def vis_rows() -> int:  return min(DD_MAX, max(0, n_cases - dd_scroll))
    def lr()       -> Rect: return Rect(hr().x, hr().bottom, hr().w, vis_rows() * ROW_H)

    def clamp_scroll() -> None:
        nonlocal dd_scroll
        dd_scroll = int(np.clip(dd_scroll, 0, max(0, n_cases - DD_MAX)))

    def ensure_visible() -> None:
        nonlocal dd_scroll
        if case_idx < dd_scroll:
            dd_scroll = case_idx
        elif case_idx >= dd_scroll + DD_MAX:
            dd_scroll = case_idx - DD_MAX + 1
        clamp_scroll()

    case_idx = 0
    ax, ay = -0.55, 0.35
    dist = 220.0
    dragging, last_mouse = False, None

    verts = faces = colors = None
    box_c = np.zeros((0, 3))
    case_name, has_box = "", False
    fov = 420.0

    def load(idx: int) -> None:
        nonlocal verts, faces, colors, box_c, case_name, has_box
        scene = load_scene(case_dirs[idx], repair_cases_dir)
        v, f, c, center = build_mesh(scene)
        verts, faces, colors = v, f, c
        case_name = case_dirs[idx].name
        bounds = _mask_bounds(scene.mask)
        if bounds:
            top, left, bottom, right = bounds
            roof  = float(scene.height[top:bottom+1, left:right+1].max()) + 2.0
            floor = float(min(scene.sea_y, scene.height.min())) - 2.0
            box_c = _box_corners(top, left, bottom, right, floor, roof, center)
            has_box = True
        else:
            box_c, has_box = np.zeros((0, 3)), False

    load(case_idx)

    def draw_dd() -> None:
        h_rect = hr()
        pygame.draw.rect(screen, (44,48,58), h_rect, border_radius=6)
        pygame.draw.rect(screen, (110,118,135), h_rect, 1, border_radius=6)
        label = f"{case_name}  ({case_idx+1}/{n_cases})"
        s = font.render(label, True, (235,237,245))
        screen.blit(s, (h_rect.x+dd_pad, h_rect.y+(HEADER_H-s.get_height())//2))
        if not dd_open:
            return
        l_rect = lr()
        pygame.draw.rect(screen, (36,40,50), l_rect)
        pygame.draw.rect(screen, (110,118,135), l_rect, 1)
        for row in range(vis_rows()):
            idx = dd_scroll + row
            if idx >= n_cases:
                break
            rr = Rect(l_rect.x, l_rect.y+row*ROW_H, l_rect.w, ROW_H)
            if idx == case_idx:
                pygame.draw.rect(screen, (58,72,98), rr)
            t = font_small.render(case_labels[idx], True, (220,224,232))
            screen.blit(t, (rr.x+dd_pad, rr.y+(ROW_H-t.get_height())//2))
        if n_cases > DD_MAX:
            track = Rect(l_rect.right-6, l_rect.y+2, 4, l_rect.h-4)
            pygame.draw.rect(screen, (26,28,34), track, border_radius=2)
            span = max(1, n_cases-DD_MAX)
            th = max(12, int(track.h*(DD_MAX/n_cases)))
            ty = track.y+int((track.h-th)*(dd_scroll/span))
            pygame.draw.rect(screen, (100,108,128), Rect(track.x,ty,track.w,th), border_radius=2)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE):
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_LEFTBRACKET:
                    case_idx = (case_idx - 1) % n_cases
                    load(case_idx)
                    ensure_visible()
                if ev.key == pygame.K_RIGHTBRACKET:
                    case_idx = (case_idx + 1) % n_cases
                    load(case_idx)
                    ensure_visible()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if hr().collidepoint(mx, my):
                    dd_open = not dd_open
                    if dd_open:
                        ensure_visible()
                    continue
                if dd_open:
                    if lr().collidepoint(mx, my):
                        idx = dd_scroll+(my-lr().y)//ROW_H
                        if 0 <= idx < n_cases:
                            case_idx = idx
                            load(case_idx)
                    dd_open = False
                    continue
                dragging, last_mouse = True, ev.pos
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging, last_mouse = False, None
            if ev.type == pygame.MOUSEMOTION and dragging and last_mouse:
                dx, dy = ev.pos[0]-last_mouse[0], ev.pos[1]-last_mouse[1]
                ay -= dx*0.005
                ax -= dy*0.005
                ax = float(np.clip(ax, -1.4, 1.4))
                last_mouse = ev.pos
            if ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if dd_open and lr().collidepoint(mx, my):
                    dd_scroll -= int(ev.y)
                    clamp_scroll()
                else:
                    dist = float(np.clip(dist*(0.9 if ev.y>0 else 1.1), 60, 900))

        Rm = _rot(ax, ay)
        cam_z = max(dist, 20.0)

        pv = verts @ Rm.T
        pz = pv[:, 2] + cam_z
        vis = pz > 0.5
        px = sw * 0.5 + fov * pv[:, 0] / pz
        py = sh * 0.5 - fov * pv[:, 1] / pz

        t_depth = pz[faces].mean(axis=1)
        order = np.argsort(-t_depth)

        screen.fill((22, 24, 30))

        for i in order:
            ia, ib, ic, id_ = faces[i]
            if not (vis[ia] and vis[ib] and vis[ic] and vis[id_]):
                continue
            poly = [(px[ia], py[ia]), (px[ib], py[ib]), (px[ic], py[ic]), (px[id_], py[id_])]
            col = colors[i]
            pygame.draw.polygon(screen, (int(col[0]), int(col[1]), int(col[2])), poly)

        if has_box and box_c.shape[0] == 8:
            bc  = box_c @ Rm.T
            bz  = bc[:,2] + cam_z
            bx_ = sw*0.5 + fov*bc[:,0]/bz
            by_ = sh*0.5 - fov*bc[:,1]/bz
            for a,b in BOX_EDGES:
                if bz[a]>0.4 and bz[b]>0.4:
                    pygame.draw.line(screen,(255,52,58),(bx_[a],by_[a]),(bx_[b],by_[b]),2)

        draw_dd()
        screen.blit(font.render(
            "muted = known  |  saturated = generated  |  red = mask  |  blue = sea  |  [/] prev/next",
            True,(160,165,180)),(8,sh-26))
        pygame.display.flip()
        clock.tick(60)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="3D block viewer for repair outputs.")
    p.add_argument("--cases-dir",        type=Path,  default=Path("outputs/saved_cases"))
    p.add_argument("--repair-cases-dir", type=Path, default=Path("repair_cases"))
    p.add_argument("--export-figures", type=Path, default=None, help="Optional directory for three-panel PNG exports")
    p.add_argument("--no-view", action="store_true", help="Export requested PNGs without opening the interactive viewer")
    p.add_argument("--limit", type=int, default=None, help="Optional limit for figure exports")
    p.add_argument("--panel-width", type=int, default=520, help="Width of each figure panel in pixels")
    p.add_argument("--panel-height", type=int, default=380, help="Height of each figure panel in pixels")
    args = p.parse_args(argv)

    cases = _discover_cases(args.cases_dir.resolve())
    if not cases:
        print(f"No cases under {args.cases_dir}", file=sys.stderr)
        sys.exit(1)

    if args.export_figures is not None:
        export_cases = cases[: args.limit] if args.limit is not None else cases
        written = export_figures(
            export_cases,
            args.repair_cases_dir.resolve(),
            args.export_figures.expanduser().resolve(),
            panel_size=(args.panel_width, args.panel_height),
        )
        print(f"Exported {len(written)} figure{'s' if len(written) != 1 else ''} to {args.export_figures}")

    if args.no_view:
        return

    _run(cases, args.repair_cases_dir.resolve())


def figures_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export isometric repair figures.")
    p.add_argument("--cases-dir", type=Path, default=Path("outputs/saved_cases"))
    p.add_argument("--repair-cases-dir", type=Path, default=Path("repair_cases"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--panel-width", type=int, default=520)
    p.add_argument("--panel-height", type=int, default=380)
    args = p.parse_args(argv)

    cases = _discover_cases(args.cases_dir.resolve())
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        print(f"No cases under {args.cases_dir}", file=sys.stderr)
        sys.exit(1)
    written = export_figures(
        cases,
        args.repair_cases_dir.resolve(),
        args.out_dir.expanduser().resolve(),
        panel_size=(args.panel_width, args.panel_height),
    )
    print(f"Exported {len(written)} figure{'s' if len(written) != 1 else ''} to {args.out_dir}")


if __name__ == "__main__":
    main()
