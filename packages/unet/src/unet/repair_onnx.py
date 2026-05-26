"""ONNX export helpers for deterministic terrain repair inference."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .repair_model import TerrainRepairUNet
from .repair_training import load_repair_model_from_checkpoint


class TerrainRepairONNXWrapper(nn.Module):
    """Export-friendly wrapper that mirrors ``deterministic_repair`` post-processing."""

    def __init__(self, model: TerrainRepairUNet):
        super().__init__()
        self.model = model

    def forward(
        self,
        known_height: torch.Tensor,
        prefill_height: torch.Tensor,
        mask: torch.Tensor,
        known_material: torch.Tensor,
        known_support: torch.Tensor,
        boundary_distance: torch.Tensor,
        prefill_gradients: torch.Tensor,
        prefill_laplacian: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(
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
        return height, outputs.material_logits, support


def build_onnx_metadata(
    checkpoint_payload: dict[str, object],
    tile_size: int,
    opset_version: int,
) -> dict[str, object]:
    meta = checkpoint_payload.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    num_material_classes = int(checkpoint_payload.get("num_material_classes", 17))
    return {
        "model_type": meta.get("model_type", "deterministic_repair_v2"),
        "num_material_classes": num_material_classes,
        "tile_size": tile_size,
        "height_min": meta.get("height_min"),
        "height_max": meta.get("height_max"),
        "unknown_material_index": 16,
        "prefill_iterations": 64,
        "opset_version": opset_version,
        "inputs": {
            "known_height": {"shape": [1, 1, tile_size, tile_size], "dtype": "float32"},
            "prefill_height": {"shape": [1, 1, tile_size, tile_size], "dtype": "float32"},
            "mask": {"shape": [1, 1, tile_size, tile_size], "dtype": "float32"},
            "known_material": {"shape": [1, tile_size, tile_size], "dtype": "int64"},
            "known_support": {"shape": [1, 1, tile_size, tile_size], "dtype": "float32"},
            "boundary_distance": {"shape": [1, 1, tile_size, tile_size], "dtype": "float32"},
            "prefill_gradients": {"shape": [1, 2, tile_size, tile_size], "dtype": "float32"},
            "prefill_laplacian": {"shape": [1, 1, tile_size, tile_size], "dtype": "float32"},
        },
        "outputs": {
            "height": {"shape": [1, 1, tile_size, tile_size], "dtype": "float32", "description": "Normalized repaired height in [0, 1]."},
            "material_logits": {
                "shape": [1, num_material_classes, tile_size, tile_size],
                "dtype": "float32",
                "description": "Per-class logits inside the mask; argmax outside mask is ignored.",
            },
            "support": {"shape": [1, 1, tile_size, tile_size], "dtype": "float32", "description": "Support proxy in [0, 1]."},
        },
        "preprocessing_notes": [
            "Set known_material to 16 (UNKNOWN_INDEX) inside masked regions before inference.",
            "Build prefill_height, boundary_distance, prefill_gradients, and prefill_laplacian from known_height and mask (see repair_data.py).",
            "Denormalize exported height with height_min/height_max from this metadata when writing blocks.",
        ],
    }


def export_repair_onnx(
    checkpoint: str | Path,
    output_path: str | Path,
    tile_size: int = 128,
    opset_version: int = 17,
    verify: bool = True,
) -> Path:
    """Export a repair checkpoint to ONNX plus a sidecar JSON metadata file."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, payload = load_repair_model_from_checkpoint(checkpoint, map_location="cpu")
    if not isinstance(model, TerrainRepairUNet):
        raise ValueError("ONNX export supports deterministic_repair_v2 checkpoints only.")
    model.eval()
    wrapper = TerrainRepairONNXWrapper(model)

    dummy = {
        "known_height": torch.zeros(1, 1, tile_size, tile_size, dtype=torch.float32),
        "prefill_height": torch.zeros(1, 1, tile_size, tile_size, dtype=torch.float32),
        "mask": torch.zeros(1, 1, tile_size, tile_size, dtype=torch.float32),
        "known_material": torch.zeros(1, tile_size, tile_size, dtype=torch.int64),
        "known_support": torch.zeros(1, 1, tile_size, tile_size, dtype=torch.float32),
        "boundary_distance": torch.zeros(1, 1, tile_size, tile_size, dtype=torch.float32),
        "prefill_gradients": torch.zeros(1, 2, tile_size, tile_size, dtype=torch.float32),
        "prefill_laplacian": torch.zeros(1, 1, tile_size, tile_size, dtype=torch.float32),
    }
    input_names = list(dummy.keys())
    output_names = ["height", "material_logits", "support"]

    dynamic_axes: dict[str, dict[int, str]] = {
        "known_height": {0: "batch", 2: "height", 3: "width"},
        "prefill_height": {0: "batch", 2: "height", 3: "width"},
        "mask": {0: "batch", 2: "height", 3: "width"},
        "known_material": {0: "batch", 1: "height", 2: "width"},
        "known_support": {0: "batch", 2: "height", 3: "width"},
        "boundary_distance": {0: "batch", 2: "height", 3: "width"},
        "prefill_gradients": {0: "batch", 2: "height", 3: "width"},
        "prefill_laplacian": {0: "batch", 2: "height", 3: "width"},
        "height": {0: "batch", 2: "height", 3: "width"},
        "material_logits": {0: "batch", 2: "height", 3: "width"},
        "support": {0: "batch", 2: "height", 3: "width"},
    }

    export_kwargs: dict[str, object] = {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    # Prefer the legacy ONNX exporter for plugin-friendly graphs without onnxscript.
    try:
        torch.onnx.export(
            wrapper,
            tuple(dummy[name] for name in input_names),
            str(output_path),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(
            wrapper,
            tuple(dummy[name] for name in input_names),
            str(output_path),
            **export_kwargs,
        )

    metadata_path = output_path.with_suffix(".json")
    metadata = build_onnx_metadata(payload, tile_size=tile_size, opset_version=opset_version)
    metadata["onnx_path"] = str(output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if verify:
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX export succeeded but verification requires optional deps: pip install onnx onnxruntime"
            ) from exc
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        ort_inputs = {name: dummy[name].numpy() for name in input_names}
        session.run(None, ort_inputs)

    return output_path


__all__ = [
    "TerrainRepairONNXWrapper",
    "build_onnx_metadata",
    "export_repair_onnx",
]
