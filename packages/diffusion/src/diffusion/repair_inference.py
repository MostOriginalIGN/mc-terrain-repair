"""Inference utilities for deterministic terrain repair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from exporter.visualize import heightmap_image, material_map_image, render_heightmap, render_material_map
from exporter.vocab import NUM_CLASSES, UNKNOWN_INDEX

from .repair_data import (
    build_prefill_height,
    compute_boundary_distance,
    compute_height_gradients,
    compute_laplacian,
    estimate_support_from_material,
)
from .repair_model import TerrainRepairUNet
from .repair_training import load_repair_checkpoint


def _load_array(path_str: str | Path, name: str) -> np.ndarray:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Missing {name} file: {path}. Use `make prepare-infer` first.")
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


def _compose_preview_panel(
    known_height: np.ndarray,
    known_material: np.ndarray,
    result_height: np.ndarray,
    result_material: np.ndarray,
    result_support: np.ndarray,
    mask: np.ndarray,
) -> Image.Image:
    panels = [
        ("Known Height", heightmap_image(known_height, mask=mask, upscale=4)),
        ("Known Material", material_map_image(known_material, mask=mask, upscale=4)),
        ("Generated Height", heightmap_image(result_height, mask=mask, upscale=4)),
        ("Generated Material", material_map_image(result_material, mask=mask, upscale=4)),
        ("Generated Support", _support_image(result_support, mask=mask, upscale=4)),
    ]
    tile_w = max(image.width for _, image in panels)
    tile_h = max(image.height for _, image in panels)
    header_h = 24
    gutter = 12
    columns = 2
    rows = 3
    canvas = Image.new(
        "RGB",
        (gutter * (columns + 1) + tile_w * columns, gutter * (rows + 1) + (tile_h + header_h) * rows),
        color=(242, 240, 235),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        row = index // columns
        col = index % columns
        x = gutter + col * (tile_w + gutter)
        y = gutter + row * (tile_h + header_h + gutter)
        draw.text((x, y), label, fill=(30, 30, 30))
        canvas.paste(image.convert("RGB"), (x, y + header_h))
    return canvas


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
    model = TerrainRepairUNet(num_material_classes=NUM_CLASSES).to(device)
    checkpoint_payload = load_repair_checkpoint(checkpoint, model, map_location=device)

    known_height_array = _load_array(known_height_path, "known height")
    known_material_array = _load_array(known_material_path, "known material")
    mask_array = _load_array(mask_path, "mask")
    support_path = Path(known_support_path).expanduser().resolve() if known_support_path is not None else Path(mask_path).with_name("known_support.npy")
    known_support_array = _maybe_load_array(support_path)

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
    np.save(output_dir / "height.npy", normalized_height)
    np.save(output_dir / "material.npy", material)
    np.save(output_dir / "support.npy", support)

    world_height = _maybe_denormalize(normalized_height, checkpoint_payload)
    preview_height = world_height if world_height is not None else normalized_height
    if world_height is not None:
        np.save(output_dir / "height_world.npy", world_height)

    render_heightmap(preview_height, output_dir / "height_preview.png", mask=mask_array, upscale=4)
    render_material_map(material, output_dir / "material_preview.png", mask=mask_array, upscale=4)
    _support_image(support, mask=mask_array, upscale=4).save(output_dir / "support_preview.png")
    preview_panel = _compose_preview_panel(
        known_height=known_height_array,
        known_material=known_material_array,
        result_height=preview_height,
        result_material=material,
        result_support=support,
        mask=mask_array,
    )
    preview_panel.save(output_dir / "preview_panel.png")

    return {
        "out_dir": output_dir,
        "height": output_dir / "height.npy",
        "material": output_dir / "material.npy",
        "support": output_dir / "support.npy",
        "preview": output_dir / "height_preview.png",
        "material_preview": output_dir / "material_preview.png",
        "support_preview": output_dir / "support_preview.png",
        "preview_panel": output_dir / "preview_panel.png",
        **({"height_world": output_dir / "height_world.npy"} if world_height is not None else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic terrain repair inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--known-height", required=True)
    parser.add_argument("--known-material", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--known-support", default=None)
    args = parser.parse_args()

    outputs = run_repair_job(
        checkpoint=args.checkpoint,
        known_height_path=args.known_height,
        known_material_path=args.known_material,
        mask_path=args.mask,
        out_dir=args.out_dir,
        known_support_path=args.known_support,
    )
    print(f"Saved repair outputs to {outputs['out_dir']}")


if __name__ == "__main__":
    main()


__all__ = [
    "build_repair_feature_tensors",
    "deterministic_repair",
    "run_repair_job",
]
