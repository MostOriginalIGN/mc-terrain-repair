from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from diffusion.infer_inputs import prepare_inference_inputs
from diffusion.repair_data import TerrainRepairDataset
from diffusion.repair_inference import run_repair_job, run_saved_case_jobs
from diffusion.repair_inference import main as repair_inference_main
from diffusion.repair_model import TerrainRepairUNet
from diffusion.repair_training import (
    RepairTrainingState,
    build_repair_checkpoint_meta,
    compute_repair_losses,
    evaluate_repair_cases,
    load_repair_checkpoint,
    save_repair_checkpoint,
    train_repair_step,
)
from exporter.vocab import UNKNOWN_INDEX


def _write_fake_export(export_dir: Path, width_chunks: int = 8, height_chunks: int = 8) -> None:
    for chunk_x in range(width_chunks):
        for chunk_z in range(height_chunks):
            surface = np.full((16, 16), 64 + chunk_x + chunk_z, dtype=np.int16)
            blocks = np.zeros((16, 16, 40), dtype=np.int8)
            blocks[:, :, :32] = 4
            if (chunk_x + chunk_z) % 3 == 0:
                blocks[:, :, 8:24] = 0
            blocks[:, :, 32] = (chunk_x * height_chunks + chunk_z) % 16
            np.save(export_dir / f"surface_{chunk_x}_{chunk_z}.npy", surface)
            np.save(export_dir / f"chunk_{chunk_x}_{chunk_z}.npy", blocks)


def _batch_from_sample(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) and value.ndim > 0 else value
        for key, value in sample.items()
    }


def test_repair_dataset_features_and_unknown_material(tmp_path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_fake_export(export_dir, width_chunks=9)

    dataset = TerrainRepairDataset(export_dir, tile_size=128, mask_mode="mixed", seed=5, cache_arrays=False)
    sample = dataset[0]

    assert sample["target_height"].shape == (1, 128, 128)
    assert sample["target_material"].shape == (128, 128)
    assert sample["target_support"].shape == (1, 128, 128)
    assert sample["known_support"].shape == (1, 128, 128)
    assert sample["prefill_height"].shape == (1, 128, 128)
    assert sample["boundary_distance"].shape == (1, 128, 128)
    assert sample["prefill_gradients"].shape == (2, 128, 128)
    assert sample["prefill_laplacian"].shape == (1, 128, 128)
    assert torch.isfinite(sample["prefill_height"]).all()
    assert torch.allclose(sample["prefill_height"] * (1.0 - sample["mask"]), sample["target_height"] * (1.0 - sample["mask"]))
    assert sample["known_material"][sample["mask"].squeeze(0).bool()].eq(UNKNOWN_INDEX).all()
    assert sample["target_support"].min().item() >= 0.0
    assert sample["target_support"].max().item() <= 1.0

    terrain_dataset = TerrainRepairDataset(export_dir, tile_size=128, mask_mode="terrain_mixed", seed=11, cache_arrays=False)
    epoch_zero_mask = terrain_dataset[0]["mask"]
    terrain_dataset.set_mask_epoch(1)
    epoch_one_mask = terrain_dataset[0]["mask"]
    assert epoch_zero_mask.shape == (1, 128, 128)
    assert epoch_zero_mask.sum().item() > 0
    assert epoch_one_mask.sum().item() > 0
    assert not torch.equal(epoch_zero_mask, epoch_one_mask)


def test_repair_model_training_and_checkpoint_roundtrip(tmp_path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_fake_export(export_dir, width_chunks=9)

    dataset = TerrainRepairDataset(export_dir, tile_size=128, mask_mode="rectangle", seed=9)
    batch = _batch_from_sample(dataset[0])

    model = TerrainRepairUNet(num_material_classes=dataset.num_material_classes)
    outputs = model(
        known_height=batch["known_height"],
        prefill_height=batch["prefill_height"],
        mask=batch["mask"],
        known_material=batch["known_material"],
        known_support=batch["known_support"],
        boundary_distance=batch["boundary_distance"],
        prefill_gradients=batch["prefill_gradients"],
        prefill_laplacian=batch["prefill_laplacian"],
    )
    assert outputs.height_residual.shape == (1, 1, 128, 128)
    assert outputs.material_logits.shape == (1, dataset.num_material_classes, 128, 128)
    assert outputs.support.shape == (1, 1, 128, 128)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = compute_repair_losses(model, batch)
    assert torch.isfinite(losses.total_loss)
    assert losses.total_loss.item() > 0

    first_param = next(model.parameters()).detach().clone()
    train_repair_step(model, optimizer, batch)
    assert not torch.allclose(first_param, next(model.parameters()).detach())

    accum_model = TerrainRepairUNet(num_material_classes=dataset.num_material_classes)
    accum_optimizer = torch.optim.AdamW(accum_model.parameters(), lr=1e-4)
    accum_first_param = next(accum_model.parameters()).detach().clone()
    train_repair_step(accum_model, accum_optimizer, batch, loss_scale=2, step_optimizer=False)
    assert torch.allclose(accum_first_param, next(accum_model.parameters()).detach())
    train_repair_step(accum_model, accum_optimizer, batch, loss_scale=2, step_optimizer=True, zero_grad=False)
    assert not torch.allclose(accum_first_param, next(accum_model.parameters()).detach())

    checkpoint = tmp_path / "repair.pt"
    args = type("Args", (), {
        "tile_size": 128,
        "stride_chunks": 1,
        "export_dir": str(export_dir),
    })()
    meta = build_repair_checkpoint_meta(args, dataset, RepairTrainingState(completed_epochs=2, global_step=12), interrupted=False)
    save_repair_checkpoint(checkpoint, model, optimizer, meta=meta)

    reloaded = TerrainRepairUNet(num_material_classes=dataset.num_material_classes)
    payload = load_repair_checkpoint(checkpoint, reloaded)
    assert payload["meta"]["model_type"] == "deterministic_repair_v1"
    assert payload["meta"]["epoch"] == 2
    assert payload["meta"]["global_step"] == 12

    bad_checkpoint = tmp_path / "bad_repair.pt"
    bad_checkpoint.write_bytes(b"not a torch checkpoint")
    with pytest.raises(RuntimeError, match="Could not load repair checkpoint"):
        load_repair_checkpoint(bad_checkpoint, reloaded)


def test_repair_inference_outputs_and_preserves_known_pixels(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_fake_export(export_dir)

    inputs_dir = tmp_path / "inputs"
    prepare_inference_inputs(
        export_dir=export_dir,
        out_dir=inputs_dir,
        tile_size=128,
        mask_top=32,
        mask_left=24,
        mask_height=48,
        mask_width=56,
    )

    checkpoint = tmp_path / "repair.pt"
    model = TerrainRepairUNet(num_material_classes=17)
    save_repair_checkpoint(
        checkpoint,
        model,
        optimizer=None,
        meta={"model_type": "deterministic_repair_v1", "height_min": 10.0, "height_max": 50.0},
    )

    out_dir = tmp_path / "outputs"
    outputs = run_repair_job(
        checkpoint=checkpoint,
        known_height_path=inputs_dir / "known_height.npy",
        known_material_path=inputs_dir / "known_material.npy",
        mask_path=inputs_dir / "mask.npy",
        out_dir=out_dir,
        known_support_path=inputs_dir / "known_support.npy",
    )

    assert outputs["height"].exists()
    assert outputs["material"].exists()
    assert outputs["support"].exists()
    assert outputs["mask"].exists()
    assert outputs["preview"].exists()
    assert outputs["material_preview"].exists()
    assert outputs["support_preview"].exists()
    assert outputs["preview_panel"].exists()
    assert outputs["combined_render"].exists()

    known_height = np.load(inputs_dir / "known_height.npy")
    mask = np.load(inputs_dir / "mask.npy")
    repaired_height = np.load(out_dir / "height.npy")
    assert np.allclose(repaired_height[mask == 0], known_height[mask == 0])

    saved_case = inputs_dir / "saved_cases" / "case_a"
    saved_case.mkdir(parents=True)
    for name in (
        "known_height.npy",
        "known_material.npy",
        "known_support.npy",
        "mask.npy",
        "target_height.npy",
        "target_material.npy",
        "target_support.npy",
    ):
        (saved_case / name).write_bytes((inputs_dir / name).read_bytes())
    saved_outputs = run_saved_case_jobs(
        checkpoint=checkpoint,
        saved_cases_dir=inputs_dir / "saved_cases",
        out_dir=tmp_path / "saved_outputs",
    )
    assert len(saved_outputs) == 1
    assert saved_outputs[0]["preview_panel"].exists()
    assert saved_outputs[0]["combined_render"].exists()
    assert (tmp_path / "saved_outputs" / "combined_all_cases.png").exists()

    metrics = evaluate_repair_cases(model, inputs_dir / "saved_cases", device=torch.device("cpu"))
    assert metrics is not None
    assert metrics.case_count == 1
    assert np.isfinite(metrics.score)
    assert 0.0 <= metrics.material_accuracy <= 1.0

    argv = [
        "repair_inference",
        "--checkpoint",
        str(checkpoint),
        "--skip-current",
        "--saved-cases-dir",
        str(inputs_dir / "saved_cases"),
        "--saved-cases-out-dir",
        str(tmp_path / "cli_saved_outputs"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    repair_inference_main()
    assert (tmp_path / "cli_saved_outputs" / "case_a" / "combined_render.png").exists()
