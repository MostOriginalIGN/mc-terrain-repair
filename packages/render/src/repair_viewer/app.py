"""3D block viewer for repair outputs"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


def _cell_rgb(material: np.ndarray, mask: np.ndarray, r: int, c: int) -> np.ndarray:
    mi = int(material[r, c]) % len(_VOCAB_RGB)
    base = np.array(_VOCAB_RGB[mi], dtype=np.float32)
    if mask[r, c] <= 0:
        return (base * 0.52 + np.array([70, 72, 78], dtype=np.float32)).clip(0, 255)
    return base


def build_mesh(scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H, W = scene.height.shape
    floor_y = float(scene.height.min()) - 1.0

    verts_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []
    base_v = 0

    def add_quad(corners: np.ndarray, shade: float, rgb: np.ndarray) -> None:
        nonlocal base_v
        col = np.clip(rgb * shade, 0, 255).astype(np.uint8)
        verts_list.append(corners)
        faces_list.append(np.array([[base_v, base_v+1, base_v+2],
                                     [base_v, base_v+2, base_v+3]], dtype=np.int32))
        colors_list.extend([col, col])
        base_v += 4

    for r in range(H):
        for c in range(W):
            y = float(scene.height[r, c])
            x0, x1 = float(c), float(c + 1)
            z0, z1 = float(r), float(r + 1)
            rgb = _cell_rgb(scene.material, scene.mask, r, c)

            add_quad(np.array([
                [x0, y, z0], [x1, y, z0],
                [x1, y, z1], [x0, y, z1],
            ]), _SHADE["top"], rgb)

            nc = c + 1
            ny_px = float(scene.height[r, min(nc, W-1)]) if nc < W else floor_y
            if ny_px < y:
                add_quad(np.array([
                    [x1, ny_px, z0], [x1, y,    z0],
                    [x1, y,    z1],  [x1, ny_px, z1],
                ]), _SHADE["px"], rgb)

            nc = c - 1
            ny_nx = float(scene.height[r, max(nc, 0)]) if nc >= 0 else floor_y
            if ny_nx < y:
                add_quad(np.array([
                    [x0, y,    z0], [x0, ny_nx, z0],
                    [x0, ny_nx, z1], [x0, y,    z1],
                ]), _SHADE["nx"], rgb)

            nr = r + 1
            ny_pz = float(scene.height[min(nr, H-1), c]) if nr < H else floor_y
            if ny_pz < y:
                add_quad(np.array([
                    [x0, y,    z1], [x1, y,    z1],
                    [x1, ny_pz, z1], [x0, ny_pz, z1],
                ]), _SHADE["pz"], rgb)

            nr = r - 1
            ny_nz = float(scene.height[max(nr, 0), c]) if nr >= 0 else floor_y
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
            if float(hmap[r, c]) >= sy:
                c += 1
                continue
            c0 = c
            while c < W and float(hmap[r, c]) < sy:
                c += 1
            x0, x1 = float(c0), float(c)
            z0, z1 = float(r), float(r + 1)
            add_quad(
                np.array([[x0, sy, z0], [x1, sy, z0], [x1, sy, z1], [x0, sy, z1]]),
                WATER_SHADE,
                WATER_RGB,
            )

    verts = np.concatenate(verts_list, axis=0).astype(np.float64)
    faces = np.concatenate(faces_list, axis=0)
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


BOX_EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


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
            max((font.size(f"  {l}  ")[0] for l in case_labels), default=200) + 36, 200, 360))
    def hr()       -> Rect: return Rect(dd_x, dd_y, dd_w(), HEADER_H)
    def vis_rows() -> int:  return min(DD_MAX, max(0, n_cases - dd_scroll))
    def lr()       -> Rect: return Rect(hr().x, hr().bottom, hr().w, vis_rows() * ROW_H)

    def clamp_scroll() -> None:
        nonlocal dd_scroll
        dd_scroll = int(np.clip(dd_scroll, 0, max(0, n_cases - DD_MAX)))

    def ensure_visible() -> None:
        nonlocal dd_scroll
        if case_idx < dd_scroll: dd_scroll = case_idx
        elif case_idx >= dd_scroll + DD_MAX: dd_scroll = case_idx - DD_MAX + 1
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
            t, l, b, r = bounds
            roof  = float(scene.height[t:b+1, l:r+1].max()) + 2.0
            floor = float(min(scene.sea_y, scene.height.min())) - 2.0
            box_c = _box_corners(t, l, b, r, floor, roof, center)
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
        if not dd_open: return
        l_rect = lr()
        pygame.draw.rect(screen, (36,40,50), l_rect)
        pygame.draw.rect(screen, (110,118,135), l_rect, 1)
        for row in range(vis_rows()):
            idx = dd_scroll + row
            if idx >= n_cases: break
            rr = Rect(l_rect.x, l_rect.y+row*ROW_H, l_rect.w, ROW_H)
            if idx == case_idx: pygame.draw.rect(screen, (58,72,98), rr)
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
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_LEFTBRACKET:
                    case_idx=(case_idx-1)%n_cases; load(case_idx); ensure_visible()
                if ev.key == pygame.K_RIGHTBRACKET:
                    case_idx=(case_idx+1)%n_cases; load(case_idx); ensure_visible()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if hr().collidepoint(mx, my):
                    dd_open = not dd_open
                    if dd_open: ensure_visible()
                    continue
                if dd_open:
                    if lr().collidepoint(mx, my):
                        idx = dd_scroll+(my-lr().y)//ROW_H
                        if 0 <= idx < n_cases: case_idx=idx; load(case_idx)
                    dd_open = False; continue
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
                    dd_scroll -= int(ev.y); clamp_scroll()
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
            ia, ib, ic = faces[i]
            if not (vis[ia] and vis[ib] and vis[ic]):
                continue
            poly = [(px[ia], py[ia]), (px[ib], py[ib]), (px[ic], py[ic])]
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
    args = p.parse_args(argv)

    cases = _discover_cases(args.cases_dir.resolve())
    if not cases:
        print(f"No cases under {args.cases_dir}", file=sys.stderr)
        sys.exit(1)

    _run(cases, args.repair_cases_dir.resolve())


if __name__ == "__main__":
    main()