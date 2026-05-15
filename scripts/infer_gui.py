"""Local GUI for selecting a chunk region and running deterministic terrain repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageTk

ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_SRC = ROOT / 'packages' / 'diffusion' / 'src'
EXPORTER_SRC = ROOT / 'packages' / 'exporter' / 'src'
DATASET_SRC = ROOT / 'packages' / 'dataset' / 'src'

for src_path in (str(DIFFUSION_SRC), str(EXPORTER_SRC), str(DATASET_SRC)):
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from diffusion.data import TerrainDiffusionDataset  # noqa: E402
from diffusion.infer_inputs import SelectionPlan, plan_chunk_selection, prepare_inference_inputs  # noqa: E402
from diffusion.repair_inference import run_repair_job  # noqa: E402
from exporter.visualize import heightmap_image, material_map_image  # noqa: E402


MAX_DISPLAY_SIZE = 960
MAP_GUTTER = 12
MAP_HEADER = 24
SAFE_CASE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
SEA_LEVEL_Y = 64.0


def _safe_case_name(name: str) -> str:
    safe = SAFE_CASE_NAME_RE.sub("_", name.strip()).strip("._")
    return safe or "repair_case"


def _draw_mask_box(image: Image.Image, mask: np.ndarray, outline: tuple[int, int, int] = (255, 36, 36), width: int = 2) -> Image.Image:
    bounds = np.argwhere(mask > 0)
    if bounds.size == 0:
        return image
    top = int(bounds[:, 0].min())
    left = int(bounds[:, 1].min())
    bottom = int(bounds[:, 0].max())
    right = int(bounds[:, 1].max())
    scale_x = image.width / mask.shape[1]
    scale_y = image.height / mask.shape[0]
    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    draw.rectangle(
        (
            int(round(left * scale_x)),
            int(round(top * scale_y)),
            int(round((right + 1) * scale_x)) - 1,
            int(round((bottom + 1) * scale_y)) - 1,
        ),
        outline=outline,
        width=max(width, int(round(min(scale_x, scale_y)))),
    )
    return boxed


def _height_image_with_water(
    heightmap: np.ndarray,
    mask: np.ndarray | None = None,
    upscale: int = 1,
    valid_mask: np.ndarray | None = None,
    sea_level: float | None = SEA_LEVEL_Y,
) -> Image.Image:
    if valid_mask is None:
        valid_mask = np.ones(heightmap.shape, dtype=bool)
    else:
        valid_mask = valid_mask.astype(bool)
    if valid_mask.shape != heightmap.shape:
        raise ValueError(f"valid_mask shape {valid_mask.shape} must match heightmap shape {heightmap.shape}")

    valid_heights = heightmap[valid_mask]
    fill_value = float(valid_heights.mean()) if valid_heights.size else 0.0
    render_heightmap = np.where(valid_mask, heightmap, fill_value)
    image = heightmap_image(render_heightmap, mask=None, upscale=upscale).convert("RGB")

    use_sea_overlay = sea_level is not None and valid_heights.size > 0 and float(valid_heights.max()) > 2.0
    water_mask = valid_mask & (heightmap < sea_level) if use_sea_overlay else np.zeros(heightmap.shape, dtype=bool)
    if water_mask.any():
        water_alpha = Image.fromarray((water_mask.astype(np.uint8) * 255), mode="L")
        if upscale > 1:
            water_alpha = water_alpha.resize(image.size, resample=Image.Resampling.NEAREST)
        water_layer = Image.new("RGB", image.size, (72, 145, 210))
        image = Image.composite(water_layer, image, water_alpha)
    invalid_mask = ~valid_mask
    if invalid_mask.any():
        invalid_alpha = Image.fromarray((invalid_mask.astype(np.uint8) * 255), mode="L")
        if upscale > 1:
            invalid_alpha = invalid_alpha.resize(image.size, resample=Image.Resampling.NEAREST)
        invalid_layer = Image.new("RGB", image.size, (36, 38, 40))
        image = Image.composite(invalid_layer, image, invalid_alpha)
    if mask is not None:
        image = _draw_mask_box(image, mask)
    return image


def _height_legend(height_min: float, height_max: float, width: int = 260, height: int = 42) -> Image.Image:
    gradient = np.tile(np.linspace(height_min, height_max, width, dtype=np.float32), (18, 1))
    legend = Image.new("RGB", (width, height), color=(242, 240, 235))
    legend.paste(heightmap_image(gradient, upscale=1).resize((width, 18), resample=Image.Resampling.BILINEAR), (0, 0))
    draw = ImageDraw.Draw(legend)
    draw.text((0, 22), f"{height_min:.0f}", fill=(30, 30, 30))
    draw.text((width - 42, 22), f"{height_max:.0f}", fill=(30, 30, 30))
    if height_min <= SEA_LEVEL_Y <= height_max:
        x = int(round((SEA_LEVEL_Y - height_min) / max(height_max - height_min, 1e-6) * (width - 1)))
        draw.line((x, 0, x, 19), fill=(45, 122, 255), width=2)
        draw.text((max(0, min(width - 52, x - 20)), 22), f"sea {SEA_LEVEL_Y:.0f}", fill=(72, 110, 255))
    return legend


def _compose_dual_preview(heightmap: np.ndarray, material_map: np.ndarray, mask: np.ndarray | None = None) -> Image.Image:
    left = _height_image_with_water(heightmap, mask=mask, upscale=4)
    right = material_map_image(material_map, mask=mask, upscale=4)
    gutter = 12
    header_h = 24
    legend_h = 54
    canvas = Image.new('RGB', (left.width + right.width + gutter * 3, left.height + gutter * 2 + header_h + legend_h), color=(242, 240, 235))
    draw = ImageDraw.Draw(canvas)
    legend = _height_legend(float(heightmap.min()), float(heightmap.max()))
    canvas.paste(legend, (gutter, gutter))
    draw.text((gutter + legend.width + gutter, gutter + 20), f"blue overlay: y < {int(SEA_LEVEL_Y)}", fill=(45, 90, 210))
    y0 = gutter + legend_h
    draw.text((gutter, y0), 'Height', fill=(30, 30, 30))
    draw.text((gutter * 2 + left.width, y0), 'Material', fill=(30, 30, 30))
    canvas.paste(left.convert('RGB'), (gutter, y0 + header_h))
    canvas.paste(right.convert('RGB'), (gutter * 2 + left.width, y0 + header_h))
    return canvas


def _fit_for_tk(image: Image.Image, max_size: int = 512) -> Image.Image:
    scale = min(1.0, max_size / max(image.width, image.height))
    if scale >= 1.0:
        return image
    return image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        resample=Image.Resampling.NEAREST,
    )

def _build_selector_maps(dataset: TerrainDiffusionDataset) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int], tuple[float, float]]:
    coords = sorted(dataset.surface_paths)
    if not coords:
        raise SystemExit(f'No exported chunks found in {dataset.export_dir}')
    min_chunk_x = min(chunk_x for chunk_x, _ in coords)
    max_chunk_x = max(chunk_x for chunk_x, _ in coords)
    min_chunk_z = min(chunk_z for _, chunk_z in coords)
    max_chunk_z = max(chunk_z for _, chunk_z in coords)
    width_chunks = max_chunk_x - min_chunk_x + 1
    height_chunks = max_chunk_z - min_chunk_z + 1
    surface_window = np.zeros((height_chunks * 16, width_chunks * 16), dtype=np.float32)
    material_window = np.full((height_chunks * 16, width_chunks * 16), 16, dtype=np.int64)
    valid_window = np.zeros((height_chunks * 16, width_chunks * 16), dtype=bool)

    for chunk_x, chunk_z in coords:
        row = (chunk_z - min_chunk_z) * 16
        col = (chunk_x - min_chunk_x) * 16
        surface_tile = dataset._load_surface((chunk_x, chunk_z)).T.astype(np.float32)
        chunk_tile = dataset._load_chunk((chunk_x, chunk_z))[:, :, 32].T.astype(np.int64)
        surface_window[row:row + 16, col:col + 16] = surface_tile
        material_window[row:row + 16, col:col + 16] = chunk_tile
        valid_window[row:row + 16, col:col + 16] = True

    return (
        _height_image_with_water(surface_window, valid_mask=valid_window),
        material_map_image(material_window),
        (min_chunk_x, min_chunk_z, max_chunk_x, max_chunk_z),
        (float(surface_window[valid_window].min()), float(surface_window[valid_window].max())),
    )


def _compose_selector_image(height_image: Image.Image, material_image: Image.Image, height_range: tuple[float, float]) -> tuple[Image.Image, dict[str, tuple[int, int, int, int]]]:
    width = height_image.width + material_image.width + MAP_GUTTER * 3
    legend_h = 54
    height = max(height_image.height, material_image.height) + MAP_GUTTER * 2 + MAP_HEADER + legend_h
    canvas = Image.new('RGB', (width, height), color=(242, 240, 235))
    draw = ImageDraw.Draw(canvas)

    left_box = (MAP_GUTTER, MAP_GUTTER + MAP_HEADER + legend_h, MAP_GUTTER + height_image.width, MAP_GUTTER + MAP_HEADER + legend_h + height_image.height)
    right_x = MAP_GUTTER * 2 + height_image.width
    right_box = (right_x, MAP_GUTTER + MAP_HEADER + legend_h, right_x + material_image.width, MAP_GUTTER + MAP_HEADER + legend_h + material_image.height)

    draw.text((MAP_GUTTER, MAP_GUTTER), 'Height', fill=(30, 30, 30))
    draw.text((right_x, MAP_GUTTER), 'Material', fill=(30, 30, 30))
    legend = _height_legend(height_range[0], height_range[1])
    canvas.paste(legend, (MAP_GUTTER, MAP_GUTTER + 20))
    draw.text((MAP_GUTTER + legend.width + MAP_GUTTER, MAP_GUTTER + 40), 'sea color: possible water below y=64', fill=(72, 110, 255))
    canvas.paste(height_image.convert('RGB'), (left_box[0], left_box[1]))
    canvas.paste(material_image.convert('RGB'), (right_box[0], right_box[1]))
    return canvas, {'height': left_box, 'material': right_box}


class InferenceSelectionApp:
    def __init__(
        self,
        export_dir: Path,
        checkpoint: Path,
        inputs_dir: Path,
        repair_cases_dir: Path,
        out_dir: Path,
        tile_size: int,
    ) -> None:
        import tkinter as tk
        from tkinter import messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.export_dir = export_dir
        self.checkpoint = checkpoint
        self.inputs_dir = inputs_dir
        self.repair_cases_dir = repair_cases_dir
        self.out_dir = out_dir
        self.tile_size = tile_size
        self.root = tk.Tk()
        self.root.title('mc-terrain-diffusion: repair selection')

        self.dataset = TerrainDiffusionDataset(export_dir, tile_size=tile_size, mask_mode='none')
        height_image, material_image, bounds, height_range = _build_selector_maps(self.dataset)
        self.min_chunk_x, self.min_chunk_z, self.max_chunk_x, self.max_chunk_z = bounds
        self.base_width, self.base_height = height_image.size
        self.selector_image, self.panel_boxes = _compose_selector_image(height_image, material_image, height_range=height_range)
        self.display_scale = min(MAX_DISPLAY_SIZE / self.selector_image.width, MAX_DISPLAY_SIZE / self.selector_image.height)
        if self.display_scale >= 1.0:
            display_width = min(MAX_DISPLAY_SIZE, int(round(self.selector_image.width * self.display_scale)))
            display_height = min(MAX_DISPLAY_SIZE, int(round(self.selector_image.height * self.display_scale)))
        else:
            display_width = max(1, int(round(self.selector_image.width * self.display_scale)))
            display_height = max(1, int(round(self.selector_image.height * self.display_scale)))
        self.display_size = (display_width, display_height)
        self.display_image = self.selector_image.resize(self.display_size, resample=Image.Resampling.NEAREST)
        self.photo_image = ImageTk.PhotoImage(self.display_image, master=self.root)
        self.selection_start: tuple[int, int] | None = None
        self.selection_end: tuple[int, int] | None = None
        self.plan: SelectionPlan | None = None

        self.status_var = tk.StringVar(value='Drag over chunks to choose what to regenerate.')
        self.selection_var = tk.StringVar(value='Selection: none')
        self.window_var = tk.StringVar(value=f'Inference window: {self.dataset.chunks_per_side}x{self.dataset.chunks_per_side} chunks')

        frame = tk.Frame(self.root)
        frame.pack(fill='both', expand=True, padx=12, pady=12)

        self.canvas = tk.Canvas(frame, width=self.display_size[0], height=self.display_size[1], highlightthickness=1)
        self.canvas.grid(row=0, column=0, rowspan=6, sticky='nsew')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo_image)
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

        panel = tk.Frame(frame)
        panel.grid(row=0, column=1, sticky='n', padx=(12, 0))
        tk.Label(panel, text=f'Export dir: {self.export_dir}', justify='left', wraplength=360).pack(anchor='w')
        tk.Label(panel, textvariable=self.selection_var, justify='left', wraplength=360).pack(anchor='w', pady=(12, 0))
        tk.Label(panel, textvariable=self.window_var, justify='left', wraplength=360).pack(anchor='w', pady=(8, 0))
        tk.Label(panel, textvariable=self.status_var, justify='left', wraplength=360).pack(anchor='w', pady=(8, 12))
        self.run_button = tk.Button(panel, text='Run repair', command=self._run_generation)
        self.run_button.pack(anchor='w')
        self.save_button = tk.Button(panel, text='Save selected case', command=self._save_selected_case)
        self.save_button.pack(anchor='w', pady=(8, 0))

        self.preview_label = tk.Label(panel)
        self.preview_label.pack(anchor='w', pady=(12, 0))

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.selection_rects: list[int] = []
        self.window_rects: list[int] = []

    def _chunk_from_canvas(self, canvas_x: int, canvas_y: int) -> tuple[int, int]:
        scaled_boxes = {}
        for name, (left, top, right, bottom) in self.panel_boxes.items():
            scaled_boxes[name] = (
                int(round(left * self.display_size[0] / self.selector_image.width)),
                int(round(top * self.display_size[1] / self.selector_image.height)),
                int(round(right * self.display_size[0] / self.selector_image.width)),
                int(round(bottom * self.display_size[1] / self.selector_image.height)),
            )

        if scaled_boxes['height'][0] <= canvas_x < scaled_boxes['height'][2]:
            panel = scaled_boxes['height']
        elif scaled_boxes['material'][0] <= canvas_x < scaled_boxes['material'][2]:
            panel = scaled_boxes['material']
        else:
            panel = scaled_boxes['height'] if canvas_x < scaled_boxes['material'][0] else scaled_boxes['material']

        rel_x = min(max(canvas_x, panel[0]), panel[2] - 1)
        rel_y = min(max(canvas_y, panel[1]), panel[3] - 1)
        grid_x = min(self.base_width - 1, max(0, int((rel_x - panel[0]) * self.base_width / max(1, panel[2] - panel[0]))))
        grid_y = min(self.base_height - 1, max(0, int((rel_y - panel[1]) * self.base_height / max(1, panel[3] - panel[1]))))
        return self.min_chunk_x + (grid_x // 16), self.min_chunk_z + (grid_y // 16)

    def _chunk_box_to_canvas(self, min_chunk_x: int, min_chunk_z: int, max_chunk_x: int, max_chunk_z: int, panel: str) -> tuple[int, int, int, int]:
        panel_left, panel_top, panel_right, panel_bottom = self.panel_boxes[panel]
        left_px = (min_chunk_x - self.min_chunk_x) * 16
        top_px = (min_chunk_z - self.min_chunk_z) * 16
        right_px = (max_chunk_x - self.min_chunk_x + 1) * 16
        bottom_px = (max_chunk_z - self.min_chunk_z + 1) * 16
        left = int(round((panel_left + left_px) * self.display_size[0] / self.selector_image.width))
        top = int(round((panel_top + top_px) * self.display_size[1] / self.selector_image.height))
        right = int(round((panel_left + right_px) * self.display_size[0] / self.selector_image.width))
        bottom = int(round((panel_top + bottom_px) * self.display_size[1] / self.selector_image.height))
        return left, top, right, bottom

    def _update_selection(self) -> None:
        if self.selection_start is None or self.selection_end is None:
            return
        min_chunk_x = min(self.selection_start[0], self.selection_end[0])
        max_chunk_x = max(self.selection_start[0], self.selection_end[0])
        min_chunk_z = min(self.selection_start[1], self.selection_end[1])
        max_chunk_z = max(self.selection_start[1], self.selection_end[1])
        width = max_chunk_x - min_chunk_x + 1
        height = max_chunk_z - min_chunk_z + 1
        self.selection_var.set(f'Selection: chunks ({min_chunk_x}, {min_chunk_z}) to ({max_chunk_x}, {max_chunk_z}) [{width}x{height}]')

        try:
            self.plan = plan_chunk_selection(
                self.dataset.window_origins,
                self.dataset.chunks_per_side,
                min_chunk_x,
                min_chunk_z,
                max_chunk_x,
                max_chunk_z,
            )
            self.window_var.set(
                'Inference window: '
                f'origin ({self.plan.origin_chunk_x}, {self.plan.origin_chunk_z}), '
                f'mask px left={self.plan.mask_left} top={self.plan.mask_top} '
                f'size={self.plan.mask_width}x{self.plan.mask_height}'
            )
            self.status_var.set('Selection is valid. Click “Run repair” to prepare inputs and run U-Net repair.')
        except ValueError as exc:
            self.plan = None
            self.window_var.set(f'Inference window: invalid selection for tile size {self.tile_size}')
            self.status_var.set(str(exc))
        self._draw_overlays(min_chunk_x, min_chunk_z, max_chunk_x, max_chunk_z)
        self._update_preview_panel()


    def _update_preview_panel(self) -> None:
        if self.plan is None:
            self.preview_label.configure(image='', text='')
            self.preview_label.image = None
            return
        surface = self.dataset._assemble_surface_window(self.plan.origin_chunk_x, self.plan.origin_chunk_z)
        material_map = self.dataset._assemble_material_window(self.plan.origin_chunk_x, self.plan.origin_chunk_z)
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        top = self.plan.mask_top
        left = self.plan.mask_left
        bottom = top + self.plan.mask_height
        right = left + self.plan.mask_width
        mask[top:bottom, left:right] = 1.0
        preview = _fit_for_tk(_compose_dual_preview(surface, material_map, mask=mask))
        photo = ImageTk.PhotoImage(preview, master=self.root)
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

    def _draw_overlays(self, min_chunk_x: int, min_chunk_z: int, max_chunk_x: int, max_chunk_z: int) -> None:
        for rect_id in self.selection_rects:
            self.canvas.delete(rect_id)
        for rect_id in self.window_rects:
            self.canvas.delete(rect_id)
        self.selection_rects = []
        self.window_rects = []
        for panel in ('height', 'material'):
            self.selection_rects.append(
                self.canvas.create_rectangle(
                    *self._chunk_box_to_canvas(min_chunk_x, min_chunk_z, max_chunk_x, max_chunk_z, panel),
                    outline='#ff3030',
                    width=2,
                )
            )
        if self.plan is not None:
            window_max_x = self.plan.origin_chunk_x + self.dataset.chunks_per_side - 1
            window_max_z = self.plan.origin_chunk_z + self.dataset.chunks_per_side - 1
            for panel in ('height', 'material'):
                self.window_rects.append(
                    self.canvas.create_rectangle(
                        *self._chunk_box_to_canvas(self.plan.origin_chunk_x, self.plan.origin_chunk_z, window_max_x, window_max_z, panel),
                        outline='#2d73ff',
                        width=2,
                        dash=(4, 3),
                    )
                )

    def _on_press(self, event) -> None:
        self.selection_start = self._chunk_from_canvas(event.x, event.y)
        self.selection_end = self.selection_start
        self._update_selection()

    def _on_drag(self, event) -> None:
        if self.selection_start is None:
            return
        self.selection_end = self._chunk_from_canvas(event.x, event.y)
        self._update_selection()

    def _on_release(self, event) -> None:
        if self.selection_start is None:
            return
        self.selection_end = self._chunk_from_canvas(event.x, event.y)
        self._update_selection()

    def _show_preview(self, preview_path: Path) -> None:
        preview = _fit_for_tk(Image.open(preview_path), max_size=900)
        photo = ImageTk.PhotoImage(preview, master=self.root)
        window = self.tk.Toplevel(self.root)
        window.title('Regeneration preview')
        label = self.tk.Label(window, image=photo)
        label.image = photo
        label.pack(padx=8, pady=8)

    def _case_dir_for_selection(self, name: str | None = None) -> Path:
        if self.plan is None:
            raise ValueError('Choose a valid chunk selection first.')
        case_name = _safe_case_name(
            name
            or (
                f"chunks_{self.plan.selected_min_chunk_x}_{self.plan.selected_min_chunk_z}"
                f"_to_{self.plan.selected_max_chunk_x}_{self.plan.selected_max_chunk_z}"
            )
        )
        return self.repair_cases_dir / case_name

    def _prepare_selection_inputs(self, out_dir: Path) -> dict[str, object]:
        if self.plan is None:
            raise ValueError('Choose a valid chunk selection first.')
        metadata = prepare_inference_inputs(
            export_dir=self.export_dir,
            out_dir=out_dir,
            checkpoint=self.checkpoint,
            tile_size=self.tile_size,
            origin_chunk_x=self.plan.origin_chunk_x,
            origin_chunk_z=self.plan.origin_chunk_z,
            mask_top=self.plan.mask_top,
            mask_left=self.plan.mask_left,
            mask_height=self.plan.mask_height,
            mask_width=self.plan.mask_width,
        )
        case_metadata = {
            **metadata,
            'selected_min_chunk_x': self.plan.selected_min_chunk_x,
            'selected_min_chunk_z': self.plan.selected_min_chunk_z,
            'selected_max_chunk_x': self.plan.selected_max_chunk_x,
            'selected_max_chunk_z': self.plan.selected_max_chunk_z,
        }
        (out_dir / 'case.json').write_text(json.dumps(case_metadata, indent=2) + '\n', encoding='utf-8')
        return case_metadata

    def _save_selected_case(self) -> None:
        if self.plan is None:
            self.messagebox.showerror('Selection required', 'Choose a valid chunk selection first.')
            return
        from tkinter import simpledialog

        default_name = (
            f"chunks_{self.plan.selected_min_chunk_x}_{self.plan.selected_min_chunk_z}"
            f"_to_{self.plan.selected_max_chunk_x}_{self.plan.selected_max_chunk_z}"
        )
        case_name = simpledialog.askstring('Save repair case', 'Case name:', initialvalue=default_name, parent=self.root)
        if case_name is None:
            return
        case_dir = self._case_dir_for_selection(case_name)
        try:
            self._prepare_selection_inputs(case_dir)
        except Exception as exc:
            self.messagebox.showerror('Save failed', str(exc))
            return
        self.status_var.set(f'Saved repair case: {case_dir}')
        self.messagebox.showinfo('Saved', f'Saved repair case inputs to:\n{case_dir}')

    def _run_generation(self) -> None:
        if self.plan is None:
            self.messagebox.showerror('Selection required', 'Choose a valid chunk selection first.')
            return
        self.run_button.config(state='disabled')
        self.status_var.set('Preparing inference inputs...')
        self.root.update_idletasks()
        try:
            self._prepare_selection_inputs(self.inputs_dir)
            self.status_var.set('Running U-Net repair...')
            self.root.update_idletasks()
            outputs = run_repair_job(
                checkpoint=self.checkpoint,
                known_height_path=self.inputs_dir / 'known_height.npy',
                known_material_path=self.inputs_dir / 'known_material.npy',
                mask_path=self.inputs_dir / 'mask.npy',
                out_dir=self.out_dir,
                known_support_path=self.inputs_dir / 'known_support.npy',
            )
        except Exception as exc:
            self.status_var.set('Repair failed.')
            self.messagebox.showerror('Repair failed', str(exc))
        else:
            preview_path = outputs.get('preview_panel', outputs['preview'])
            self.status_var.set(f'Repair complete: {preview_path}')
            self._show_preview(preview_path)
            self.messagebox.showinfo('Done', f'Repair finished. Outputs written to {outputs["out_dir"]}.')
        finally:
            self.run_button.config(state='normal')

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description='Select a chunk region in a local GUI and run U-Net terrain repair.')
    parser.add_argument('--export-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--inputs-dir', required=True)
    parser.add_argument('--repair-cases-dir', default='./repair_cases')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--tile-size', type=int, default=128)
    args = parser.parse_args()

    app = InferenceSelectionApp(
        export_dir=Path(args.export_dir).expanduser().resolve(),
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        inputs_dir=Path(args.inputs_dir).expanduser().resolve(),
        repair_cases_dir=Path(args.repair_cases_dir).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        tile_size=args.tile_size,
    )
    app.run()


if __name__ == '__main__':
    main()
