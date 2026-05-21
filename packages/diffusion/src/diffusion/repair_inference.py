"""Inference utilities for deterministic terrain repair."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from exporter.visualize import heightmap_image, material_map_image, render_heightmap, render_material_map
from exporter.vocab import UNKNOWN_INDEX

from .repair_data import (
    build_prefill_height,
    compute_boundary_distance,
    compute_height_gradients,
    compute_laplacian,
    estimate_support_from_material,
)
from .repair_model import TerrainRepairUNet
from .repair_training import load_repair_model_from_checkpoint

SEA_LEVEL_Y = 64.0


def _load_array(path_str: str | Path, name: str) -> np.ndarray:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Missing {name} file: {path}. Use `make repair` for shared cases or `make infer` after preparing scratch inputs.")
    return np.load(path)


def _maybe_load_array(path: Path) -> np.ndarray | None:
    return np.load(path) if path.is_file() else None


def _maybe_denormalize(height: np.ndarray, checkpoint_payload: dict[str, object]) -> np.ndarray | None:
    meta = checkpoint_payload.get("meta")
    if not isinstance(meta, dict):
        return None
    height_min = meta.get("height_min")
    height_max = meta.get("height_max")
    if not isinstance(height_min, (int, float)) or not isinstance(height_max, (int, float)):
        return None
    return height * (float(height_max) - float(height_min)) + float(height_min)


def _denormalize_with_range(height: np.ndarray, height_range: tuple[float, float] | None) -> np.ndarray:
    if height_range is None:
        return height
    height_min, height_max = height_range
    return height * (height_max - height_min) + height_min


def _height_range(checkpoint_payload: dict[str, object]) -> tuple[float, float] | None:
    meta = checkpoint_payload.get("meta")
    if not isinstance(meta, dict):
        return None
    height_min = meta.get("height_min")
    height_max = meta.get("height_max")
    if isinstance(height_min, (int, float)) and isinstance(height_max, (int, float)):
        return float(height_min), float(height_max)
    return None


def _height_image_with_water(
    height: np.ndarray,
    mask: np.ndarray | None,
    upscale: int,
    sea_level: float | None = SEA_LEVEL_Y,
) -> Image.Image:
    image = heightmap_image(height, mask=None, upscale=upscale).convert("RGB")
    if sea_level is not None:
        water_mask = height < sea_level
        if water_mask.any():
            water = Image.fromarray((water_mask.astype(np.uint8) * 255), mode="L")
            if upscale > 1:
                water = water.resize(image.size, resample=Image.Resampling.NEAREST)
            water_layer = Image.new("RGB", image.size, (72, 145, 210))
            image = Image.composite(water_layer, image, water)
    if mask is not None:
        image = _draw_mask_box(image, mask)
    return image


def _height_legend(height_min: float, height_max: float, width: int = 260, height: int = 42, sea_level: float = SEA_LEVEL_Y) -> Image.Image:
    gradient = np.tile(np.linspace(height_min, height_max, width, dtype=np.float32), (18, 1))
    legend = Image.new("RGB", (width, height), color=(242, 240, 235))
    legend.paste(heightmap_image(gradient, upscale=1).resize((width, 18), resample=Image.Resampling.BILINEAR), (0, 0))
    draw = ImageDraw.Draw(legend)
    draw.text((0, 22), f"{height_min:.0f}", fill=(30, 30, 30))
    draw.text((width - 42, 22), f"{height_max:.0f}", fill=(30, 30, 30))
    if height_min <= sea_level <= height_max:
        x = int(round((sea_level - height_min) / max(height_max - height_min, 1e-6) * (width - 1)))
        draw.line((x, 0, x, 19), fill=(45, 122, 255), width=2)
        draw.text((max(0, min(width - 52, x - 20)), 22), f"sea {sea_level:.0f}", fill=(72, 110, 255))
    return legend


def _support_image(support: np.ndarray, mask: np.ndarray | None = None, upscale: int = 4) -> Image.Image:
    support_uint8 = (support.clip(0.0, 1.0) * 255).astype(np.uint8)
    image = Image.fromarray(support_uint8, mode="L").convert("RGB")
    if mask is not None:
        bounds = np.argwhere(mask > 0)
        if bounds.size > 0:
            top = int(bounds[:, 0].min())
            left = int(bounds[:, 1].min())
            bottom = int(bounds[:, 0].max())
            right = int(bounds[:, 1].max())
            draw = ImageDraw.Draw(image)
            draw.rectangle((left, top, right, bottom), outline=(255, 36, 36), width=2)
    if upscale > 1:
        image = image.resize((image.width * upscale, image.height * upscale), resample=Image.Resampling.NEAREST)
    return image


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


def _height_error_image(
    predicted_height: np.ndarray,
    target_height: np.ndarray,
    mask: np.ndarray | None = None,
    upscale: int = 4,
) -> Image.Image:
    error = np.abs(predicted_height.astype(np.float32) - target_height.astype(np.float32))
    if mask is not None and np.any(mask > 0):
        visible_error = error[mask > 0]
        high = float(visible_error.max()) if visible_error.size else float(error.max())
    else:
        high = float(error.max())
    normalized = (error / max(high, 1e-6)).clip(0.0, 1.0)
    image = np.zeros(error.shape + (3,), dtype=np.uint8)
    image[..., 0] = (normalized * 255).astype(np.uint8)
    image[..., 1] = (70 * (1.0 - normalized)).astype(np.uint8)
    image[..., 2] = (255 * (1.0 - normalized)).astype(np.uint8)
    rendered = Image.fromarray(image, mode="RGB")
    if mask is not None:
        rendered = _draw_mask_box(rendered, mask)
    if upscale > 1:
        rendered = rendered.resize((rendered.width * upscale, rendered.height * upscale), resample=Image.Resampling.NEAREST)
    return rendered


def _compose_preview_panel(
    known_height: np.ndarray,
    known_material: np.ndarray,
    known_support: np.ndarray,
    result_height: np.ndarray,
    result_material: np.ndarray,
    result_support: np.ndarray,
    mask: np.ndarray,
    height_range: tuple[float, float] | None = None,
    target_height: np.ndarray | None = None,
    target_material: np.ndarray | None = None,
    target_support: np.ndarray | None = None,
) -> Image.Image:
    panels = [
        ("Known Height", _height_image_with_water(known_height, mask=mask, upscale=4, sea_level=SEA_LEVEL_Y if height_range else None)),
        ("Known Material", material_map_image(known_material, mask=mask, upscale=4)),
        ("Known Support", _support_image(known_support, mask=mask, upscale=4)),
    ]
    if target_height is not None:
        panels.append(("True Height", _height_image_with_water(target_height, mask=mask, upscale=4, sea_level=SEA_LEVEL_Y if height_range else None)))
    if target_material is not None:
        panels.append(("True Material", material_map_image(target_material, mask=mask, upscale=4)))
    if target_support is not None:
        panels.append(("True Support", _support_image(target_support, mask=mask, upscale=4)))
    panels.extend(
        [
            ("Repaired Height", _height_image_with_water(result_height, mask=mask, upscale=4, sea_level=SEA_LEVEL_Y if height_range else None)),
            ("Repaired Material", material_map_image(result_material, mask=mask, upscale=4)),
            ("Repaired Support", _support_image(result_support, mask=mask, upscale=4)),
        ]
    )
    if target_height is not None:
        panels.append(("Masked Height Error", _height_error_image(result_height, target_height, mask=mask, upscale=4)))
    tile_w = max(image.width for _, image in panels)
    tile_h = max(image.height for _, image in panels)
    header_h = 24
    gutter = 12
    columns = 3
    rows = math.ceil(len(panels) / columns)
    legend_h = 54 if height_range is not None else 0
    canvas = Image.new(
        "RGB",
        (gutter * (columns + 1) + tile_w * columns, legend_h + gutter * (rows + 1) + (tile_h + header_h) * rows),
        color=(242, 240, 235),
    )
    draw = ImageDraw.Draw(canvas)
    y_offset = 0
    if height_range is not None:
        legend = _height_legend(height_range[0], height_range[1])
        canvas.paste(legend, (gutter, gutter))
        y_offset = legend_h
    for index, (label, image) in enumerate(panels):
        row = index // columns
        col = index % columns
        x = gutter + col * (tile_w + gutter)
        y = y_offset + gutter + row * (tile_h + header_h + gutter)
        draw.text((x, y), label, fill=(30, 30, 30))
        canvas.paste(image.convert("RGB"), (x, y + header_h))
    return canvas


def _compose_case_contact_sheet(case_panels: list[tuple[str, Path]], out_path: Path) -> Path | None:
    if not case_panels:
        return None
    loaded: list[tuple[str, Image.Image]] = []
    for label, path in case_panels:
        if path.is_file():
            loaded.append((label, Image.open(path).convert("RGB")))
    if not loaded:
        return None

    max_preview_w = 520
    prepared: list[tuple[str, Image.Image]] = []
    for label, image in loaded:
        scale = min(1.0, max_preview_w / image.width)
        if scale < 1.0:
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), resample=Image.Resampling.BILINEAR)
        prepared.append((label, image))

    columns = 2 if len(prepared) > 1 else 1
    rows = math.ceil(len(prepared) / columns)
    tile_w = max(image.width for _, image in prepared)
    tile_h = max(image.height for _, image in prepared)
    header_h = 24
    gutter = 16
    canvas = Image.new(
        "RGB",
        (gutter * (columns + 1) + tile_w * columns, gutter * (rows + 1) + (tile_h + header_h) * rows),
        color=(242, 240, 235),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(prepared):
        row = index // columns
        col = index % columns
        x = gutter + col * (tile_w + gutter)
        y = gutter + row * (tile_h + header_h + gutter)
        draw.text((x, y), label, fill=(30, 30, 30))
        canvas.paste(image, (x, y + header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def build_repair_feature_tensors(
    known_height_array: np.ndarray,
    known_material_array: np.ndarray,
    mask_array: np.ndarray,
    known_support_array: np.ndarray | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    mask = mask_array.astype(np.float32)
    known_material = known_material_array.copy()
    known_material[mask.astype(bool)] = UNKNOWN_INDEX
    known_support = (
        known_support_array.astype(np.float32)
        if known_support_array is not None
        else estimate_support_from_material(known_material_array, mask=mask)
    )
    prefill_height = build_prefill_height(known_height_array.astype(np.float32), mask)
    boundary_distance = compute_boundary_distance(mask)
    prefill_gradients = compute_height_gradients(prefill_height)
    prefill_laplacian = compute_laplacian(prefill_height)

    return {
        "known_height": torch.from_numpy(known_height_array.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device),
        "known_material": torch.from_numpy(known_material.astype(np.int64)).unsqueeze(0).to(device),
        "known_support": torch.from_numpy(known_support.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device),
        "mask": torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device),
        "prefill_height": torch.from_numpy(prefill_height).unsqueeze(0).unsqueeze(0).to(device),
        "boundary_distance": torch.from_numpy(boundary_distance).unsqueeze(0).unsqueeze(0).to(device),
        "prefill_gradients": torch.from_numpy(prefill_gradients).unsqueeze(0).to(device),
        "prefill_laplacian": torch.from_numpy(prefill_laplacian).unsqueeze(0).unsqueeze(0).to(device),
    }


def deterministic_repair(
    model: TerrainRepairUNet,
    known_height: torch.Tensor,
    known_material: torch.Tensor,
    known_support: torch.Tensor,
    mask: torch.Tensor,
    prefill_height: torch.Tensor,
    boundary_distance: torch.Tensor,
    prefill_gradients: torch.Tensor,
    prefill_laplacian: torch.Tensor,
) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        outputs = model(
            known_height=known_height,
            prefill_height=prefill_height,
            mask=mask,
            known_material=known_material,
            known_support=known_support,
            boundary_distance=boundary_distance,
            prefill_gradients=prefill_gradients,
            prefill_laplacian=prefill_laplacian,
        )
        predicted_height = (prefill_height + outputs.height_residual).clamp(0.0, 1.0)
        height = known_height * (1.0 - mask) + predicted_height * mask
        support = known_support * (1.0 - mask) + outputs.support * mask
        predicted_material = outputs.material_logits.argmax(dim=1)
        material = known_material * (1.0 - mask.squeeze(1)).long() + predicted_material * mask.squeeze(1).long()
    return {
        "height": height,
        "material_logits": outputs.material_logits,
        "material": material,
        "support": support,
    }


def run_repair_job(
    checkpoint: str | Path,
    known_height_path: str | Path,
    known_material_path: str | Path,
    mask_path: str | Path,
    out_dir: str | Path,
    known_support_path: str | Path | None = None,
) -> dict[str, Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_payload = load_repair_model_from_checkpoint(checkpoint, map_location=device)
    model = model.to(device)
    checkpoint_height_range = _height_range(checkpoint_payload)

    known_height_array = _load_array(known_height_path, "known height")
    known_material_array = _load_array(known_material_path, "known material")
    mask_array = _load_array(mask_path, "mask")
    support_path = Path(known_support_path).expanduser().resolve() if known_support_path is not None else Path(mask_path).with_name("known_support.npy")
    known_support_array = _maybe_load_array(support_path)
    mask_file = Path(mask_path).expanduser().resolve()
    target_height_array = _maybe_load_array(mask_file.with_name("target_height.npy"))
    target_material_array = _maybe_load_array(mask_file.with_name("target_material.npy"))
    target_support_array = _maybe_load_array(mask_file.with_name("target_support.npy"))

    tensors = build_repair_feature_tensors(
        known_height_array=known_height_array,
        known_material_array=known_material_array,
        mask_array=mask_array,
        known_support_array=known_support_array,
        device=device,
    )
    result = deterministic_repair(model=model, **tensors)

    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_height = result["height"].squeeze(0).squeeze(0).cpu().numpy()
    material = result["material"].squeeze(0).cpu().numpy()
    support = result["support"].squeeze(0).squeeze(0).cpu().numpy()
    preview_known_support = tensors["known_support"].squeeze(0).squeeze(0).cpu().numpy()
    np.save(output_dir / "height.npy", normalized_height)
    np.save(output_dir / "material.npy", material)
    np.save(output_dir / "support.npy", support)
    np.save(output_dir / "mask.npy", mask_array)

    world_height = _maybe_denormalize(normalized_height, checkpoint_payload)
    preview_height = world_height if world_height is not None else normalized_height
    preview_known_height = _denormalize_with_range(known_height_array, checkpoint_height_range)
    preview_target_height = _denormalize_with_range(target_height_array, checkpoint_height_range) if target_height_array is not None else None
    if world_height is not None:
        np.save(output_dir / "height_world.npy", world_height)

    render_heightmap(preview_height, output_dir / "height_preview.png", mask=mask_array, upscale=4)
    render_material_map(material, output_dir / "material_preview.png", mask=mask_array, upscale=4)
    _support_image(support, mask=mask_array, upscale=4).save(output_dir / "support_preview.png")
    preview_panel = _compose_preview_panel(
        known_height=preview_known_height,
        known_material=known_material_array,
        known_support=preview_known_support,
        result_height=preview_height,
        result_material=material,
        result_support=support,
        mask=mask_array,
        height_range=checkpoint_height_range,
        target_height=preview_target_height,
        target_material=target_material_array,
        target_support=target_support_array,
    )
    preview_panel.save(output_dir / "preview_panel.png")
    preview_panel.save(output_dir / "combined_render.png")

    return {
        "out_dir": output_dir,
        "height": output_dir / "height.npy",
        "material": output_dir / "material.npy",
        "support": output_dir / "support.npy",
        "mask": output_dir / "mask.npy",
        "preview": output_dir / "height_preview.png",
        "material_preview": output_dir / "material_preview.png",
        "support_preview": output_dir / "support_preview.png",
        "preview_panel": output_dir / "preview_panel.png",
        "combined_render": output_dir / "combined_render.png",
        **({"height_world": output_dir / "height_world.npy"} if world_height is not None else {}),
    }


def run_saved_case_jobs(
    checkpoint: str | Path,
    saved_cases_dir: str | Path,
    out_dir: str | Path,
) -> list[dict[str, Path]]:
    """Run deterministic repair for every saved GUI case directory."""
    cases_root = Path(saved_cases_dir).expanduser().resolve()
    if not cases_root.is_dir():
        return []
    outputs: list[dict[str, Path]] = []
    case_panels: list[tuple[str, Path]] = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        required = [case_dir / "known_height.npy", case_dir / "known_material.npy", case_dir / "mask.npy"]
        if not all(path.is_file() for path in required):
            continue
        output = run_repair_job(
            checkpoint=checkpoint,
            known_height_path=case_dir / "known_height.npy",
            known_material_path=case_dir / "known_material.npy",
            mask_path=case_dir / "mask.npy",
            out_dir=Path(out_dir).expanduser().resolve() / case_dir.name,
            known_support_path=case_dir / "known_support.npy",
        )
        outputs.append(output)
        case_panels.append((case_dir.name, output["combined_render"]))
    _compose_case_contact_sheet(case_panels, Path(out_dir).expanduser().resolve() / "combined_all_cases.png")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic terrain repair inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--known-height", default=None)
    parser.add_argument("--known-material", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--known-support", default=None)
    parser.add_argument("--saved-cases-dir", default=None)
    parser.add_argument("--saved-cases-out-dir", default=None)
    parser.add_argument("--skip-current", action="store_true")
    args = parser.parse_args()

    if not args.skip_current:
        missing_current_args = [
            flag
            for flag, value in (
                ("--known-height", args.known_height),
                ("--known-material", args.known_material),
                ("--mask", args.mask),
            )
            if value is None
        ]
        if missing_current_args:
            parser.error(
                "current repair requires "
                f"{', '.join(missing_current_args)}; pass --skip-current to run only --saved-cases-dir"
            )
        outputs = run_repair_job(
            checkpoint=args.checkpoint,
            known_height_path=args.known_height,
            known_material_path=args.known_material,
            mask_path=args.mask,
            out_dir=args.out_dir,
            known_support_path=args.known_support,
        )
        print(f"Saved repair outputs to {outputs['out_dir']}")
    if args.saved_cases_dir is not None:
        saved_outputs = run_saved_case_jobs(
            checkpoint=args.checkpoint,
            saved_cases_dir=args.saved_cases_dir,
            out_dir=args.saved_cases_out_dir or Path(args.out_dir) / "saved_cases",
        )
        print(f"Saved repair outputs for {len(saved_outputs)} saved cases")


if __name__ == "__main__":
    main()


__all__ = [
    "build_repair_feature_tensors",
    "deterministic_repair",
    "run_repair_job",
    "run_saved_case_jobs",
]
