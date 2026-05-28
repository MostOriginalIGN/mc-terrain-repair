from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from unet.infer_inputs import prepare_inference_inputs
from unet.repair_data import TerrainRepairDataset
from unet.repair_inference import run_repair_job, run_saved_case_jobs
from unet.repair_inference import main as repair_inference_main
from unet.repair_lightning import TerrainRepairDataModule, TerrainRepairLightningModule, split_spatial_window_indices
from unet.repair_model import TerrainRepairUNet
from unet.repair_training import (
    RepairTrainingState,
    build_repair_checkpoint_meta,
    compute_repair_losses,
    context_style_loss,
    evaluate_repair_cases,
    height_highpass,
    height_laplacian,
    load_repair_checkpoint,
    resolve_training_export_dirs,
    save_repair_checkpoint,
    train_repair_step,
    validation_overlap_warnings,
)
from exporter.vocab import UNKNOWN_INDEX


def _write_fake_export(
    export_dir: Path,
    width_chunks: int = 8,
    height_chunks: int = 8,
    *,
    height_base: int = 64,
    surface_material: int | None = None,
) -> None:
    for chunk_x in range(width_chunks):
        for chunk_z in range(height_chunks):
            surface = np.full((16, 16), height_base + chunk_x + chunk_z, dtype=np.int16)
            blocks = np.zeros((16, 16, 40), dtype=np.int8)
            blocks[:, :, :32] = 4
            if (chunk_x + chunk_z) % 3 == 0:
                blocks[:, :, 8:24] = 0
            blocks[:, :, 32] = surface_material if surface_material is not None else (chunk_x * height_chunks + chunk_z) % 16
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
    assert sample["height_scale"].shape == ()
    assert sample["height_scale"].item() > 0
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

    selection_dataset = TerrainRepairDataset(export_dir, tile_size=128, mask_mode="selection_mixed", seed=13, cache_arrays=False)
    selection_mask = selection_dataset[0]["mask"]
    selection_dataset.set_mask_epoch(1)
    next_selection_mask = selection_dataset[0]["mask"]
    assert selection_mask.shape == (1, 128, 128)
    assert 0 < selection_mask.sum().item() < 128 * 128
    assert next_selection_mask.sum().item() > 0
    assert not torch.equal(selection_mask, next_selection_mask)


def test_repair_dataset_augmentation_toggle(tmp_path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_fake_export(export_dir, width_chunks=9)

    base_dataset = TerrainRepairDataset(
        export_dir,
        tile_size=128,
        mask_mode="rectangle",
        seed=7,
        cache_arrays=False,
        augment=False,
    )
    base_a = base_dataset[0]
    base_b = base_dataset[0]
    assert torch.equal(base_a["target_height"], base_b["target_height"])
    assert torch.equal(base_a["target_material"], base_b["target_material"])
    assert torch.equal(base_a["mask"], base_b["mask"])

    augmented_dataset = TerrainRepairDataset(
        export_dir,
        tile_size=128,
        mask_mode="rectangle",
        seed=7,
        cache_arrays=False,
        augment=True,
    )
    aug_a = augmented_dataset[0]["target_height"]
    aug_b = augmented_dataset[0]["target_height"]
    assert torch.equal(aug_a, aug_b)
    augmented_dataset.set_mask_epoch(1)
    aug_epoch_one = augmented_dataset[0]["target_height"]
    assert not torch.equal(aug_a, aug_epoch_one)
    sample = augmented_dataset[0]
    assert sample["prefill_gradients"].shape == (2, 128, 128)
    assert sample["prefill_laplacian"].shape == (1, 128, 128)
    assert torch.allclose(
        sample["prefill_height"] * (1.0 - sample["mask"]),
        sample["target_height"] * (1.0 - sample["mask"]),
    )


def test_repair_dataset_keeps_multi_export_windows_separate(tmp_path) -> None:
    export_a = tmp_path / "export_a"
    export_b = tmp_path / "export_b"
    export_a.mkdir()
    export_b.mkdir()
    _write_fake_export(export_a, width_chunks=8, height_chunks=8, height_base=40, surface_material=2)
    _write_fake_export(export_b, width_chunks=8, height_chunks=8, height_base=90, surface_material=9)

    dataset = TerrainRepairDataset([export_a, export_b], tile_size=128, mask_mode="none", cache_arrays=False)

    assert len(dataset) == 2
    sample_a = dataset[0]
    sample_b = dataset[1]
    assert sample_a["target_material"].unique().tolist() == [2]
    assert sample_b["target_material"].unique().tolist() == [9]
    assert sample_a["target_height"].mean().item() < sample_b["target_height"].mean().item()


def test_resolve_training_export_dirs_from_parent_directory(tmp_path) -> None:
    exports_root = tmp_path / "exports"
    export_a = exports_root / "world_a"
    export_b = exports_root / "world_b"
    export_c = exports_root / "notes"
    export_a.mkdir(parents=True)
    export_b.mkdir(parents=True)
    export_c.mkdir(parents=True)
    _write_fake_export(export_a, width_chunks=8, height_chunks=8, height_base=50)
    _write_fake_export(export_b, width_chunks=8, height_chunks=8, height_base=80)
    (export_c / "readme.txt").write_text("not an export", encoding="utf-8")

    resolved = resolve_training_export_dirs([str(exports_root)])

    assert resolved == [export_a.resolve(), export_b.resolve()]


def test_validation_overlap_warning_detects_shared_export_dir(tmp_path) -> None:
    export_dir = tmp_path / "export"
    cases_dir = tmp_path / "cases"
    case_dir = cases_dir / "case_a"
    export_dir.mkdir()
    case_dir.mkdir(parents=True)
    (case_dir / "metadata.json").write_text(
        json.dumps({"export_dir": str(export_dir), "origin_chunk_x": 0, "origin_chunk_z": 0}),
        encoding="utf-8",
    )

    warnings = validation_overlap_warnings(cases_dir, [export_dir])

    assert len(warnings) == 1
    assert "case_a" in warnings[0]


def test_spatial_split_holds_out_non_overlapping_windows(tmp_path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_fake_export(export_dir, width_chunks=24, height_chunks=12)
    dataset = TerrainRepairDataset(export_dir, tile_size=128, stride_chunks=1, mask_mode="rectangle", cache_arrays=False)

    train_indices, val_indices = split_spatial_window_indices(dataset, val_fraction=0.25, buffer_chunks=4)

    assert train_indices
    assert val_indices
    train_origins = [dataset.window_origins[index] for index in train_indices]
    val_origins = [dataset.window_origins[index] for index in val_indices]
    min_val_x = min(origin[0] for origin in val_origins)
    max_train_x = max(origin[0] for origin in train_origins)
    assert max_train_x + dataset.chunks_per_side - 1 + 4 < min_val_x


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
    assert torch.isfinite(losses.laplacian_loss)
    assert torch.isfinite(losses.highpass_loss)
    assert torch.isfinite(losses.roughness_loss)
    assert torch.isfinite(losses.context_loss)
    assert torch.isfinite(losses.height_mae_blocks)
    assert 0.0 <= losses.height_within_1_block.item() <= 1.0
    assert 0.0 <= losses.height_within_2_blocks.item() <= 1.0
    assert torch.isfinite(losses.gradient_mae_blocks)

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
        "export_dir": [str(export_dir)],
    })()
    meta = build_repair_checkpoint_meta(args, dataset, RepairTrainingState(completed_epochs=2, global_step=12), interrupted=False)
    save_repair_checkpoint(checkpoint, model, optimizer, meta=meta)

    reloaded = TerrainRepairUNet(num_material_classes=dataset.num_material_classes)
    payload = load_repair_checkpoint(checkpoint, reloaded)
    assert payload["meta"]["model_type"] == "deterministic_repair_v2"
    assert payload["meta"]["model_depth"] == 4
    assert payload["meta"]["epoch"] == 2
    assert payload["meta"]["global_step"] == 12
    assert payload["meta"]["export_dirs"] == [str(export_dir.resolve())]
    assert payload["meta"]["validation_score_type"] == "visual_score"
    assert payload["meta"]["loss_weights"]["material"] < 0.2

    bad_checkpoint = tmp_path / "bad_repair.pt"
    bad_checkpoint.write_bytes(b"not a torch checkpoint")
    with pytest.raises(RuntimeError, match="Could not load repair checkpoint"):
        load_repair_checkpoint(bad_checkpoint, reloaded)


def test_repair_model_dropout_train_vs_eval(tmp_path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_fake_export(export_dir, width_chunks=9)
    dataset = TerrainRepairDataset(export_dir, tile_size=128, mask_mode="rectangle", seed=3)
    batch = _batch_from_sample(dataset[0])

    model = TerrainRepairUNet(num_material_classes=dataset.num_material_classes, dropout=0.1)
    model.train()
    torch.manual_seed(0)
    train_out = model(
        known_height=batch["known_height"],
        prefill_height=batch["prefill_height"],
        mask=batch["mask"],
        known_material=batch["known_material"],
        known_support=batch["known_support"],
        boundary_distance=batch["boundary_distance"],
        prefill_gradients=batch["prefill_gradients"],
        prefill_laplacian=batch["prefill_laplacian"],
    )
    model.eval()
    torch.manual_seed(0)
    eval_out = model(
        known_height=batch["known_height"],
        prefill_height=batch["prefill_height"],
        mask=batch["mask"],
        known_material=batch["known_material"],
        known_support=batch["known_support"],
        boundary_distance=batch["boundary_distance"],
        prefill_gradients=batch["prefill_gradients"],
        prefill_laplacian=batch["prefill_laplacian"],
    )
    assert not torch.allclose(train_out.height_residual, eval_out.height_residual)


def test_height_detail_operators_detect_oversmoothing() -> None:
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    target = (((xx // 4 + yy // 4) % 2).float() * 0.2 + yy.float() / 64.0).view(1, 1, 32, 32)
    smooth = torch.nn.functional.avg_pool2d(target, kernel_size=7, stride=1, padding=3, count_include_pad=False)

    lap_error = (height_laplacian(smooth) - height_laplacian(target)).abs().mean()
    highpass_error = (height_highpass(smooth) - height_highpass(target)).abs().mean()

    assert lap_error.item() > 0.01
    assert highpass_error.item() > 0.01


def test_context_style_loss_detects_mismatched_local_terrain() -> None:
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    target = (((xx // 3 + yy // 3) % 2).float() * 0.16 + yy.float() / 96.0).view(1, 1, 64, 64)
    mask = torch.zeros_like(target)
    mask[:, :, 20:44, 20:44] = 1.0
    smooth_inside = torch.nn.functional.avg_pool2d(target, kernel_size=11, stride=1, padding=5, count_include_pad=False)
    composite = target * (1.0 - mask) + smooth_inside * mask

    matched_loss = context_style_loss(target, target, mask)
    smoothed_loss = context_style_loss(composite, target, mask)

    assert smoothed_loss.item() > matched_loss.item() + 0.01


def test_lightning_module_training_step(tmp_path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_fake_export(export_dir, width_chunks=8, height_chunks=8)

    datamodule = TerrainRepairDataModule(
        export_dirs=[export_dir],
        tile_size=128,
        stride_chunks=1,
        mask_mode="rectangle",
        augment=False,
        batch_size=1,
        num_workers=0,
    )
    datamodule.setup("fit")
    assert datamodule.dataset is not None

    batch = _batch_from_sample(datamodule.dataset[0])
    module = TerrainRepairLightningModule(num_material_classes=datamodule.dataset.num_material_classes)
    loss = module.training_step(batch, batch_idx=0)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_lightning_optimizer_uses_weight_decay(tmp_path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_fake_export(export_dir, width_chunks=8, height_chunks=8)
    dataset = TerrainRepairDataset(export_dir, tile_size=128, mask_mode="rectangle")
    module = TerrainRepairLightningModule(
        num_material_classes=dataset.num_material_classes,
        weight_decay=0.123,
        dropout=0.1,
    )
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.123)


def test_repair_onnx_export_roundtrip(tmp_path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from unet.repair_onnx import export_repair_onnx

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_fake_export(export_dir, width_chunks=8, height_chunks=8)
    dataset = TerrainRepairDataset(export_dir, tile_size=128, mask_mode="rectangle", seed=1)

    model = TerrainRepairUNet(num_material_classes=dataset.num_material_classes, dropout=0.0)
    checkpoint = tmp_path / "repair.pt"
    save_repair_checkpoint(
        checkpoint,
        model,
        optimizer=None,
        meta={"model_type": "deterministic_repair_v2", "height_min": 0.0, "height_max": 128.0},
    )

    onnx_path = tmp_path / "repair.onnx"
    export_repair_onnx(checkpoint, onnx_path, tile_size=128, verify=True)
    assert onnx_path.is_file()
    assert onnx_path.with_suffix(".json").is_file()


def test_checkpoint_meta_includes_regularization_fields(tmp_path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_fake_export(export_dir, width_chunks=8, height_chunks=8)
    dataset = TerrainRepairDataset(export_dir, tile_size=128, mask_mode="rectangle")
    args = type("Args", (), {
        "tile_size": 128,
        "stride_chunks": 1,
        "export_dir": [str(export_dir)],
        "dropout": 0.1,
        "weight_decay": 1e-2,
        "augment": True,
        "lr_scheduler": "cosine",
        "learning_rate": 1e-4,
    })()
    meta = build_repair_checkpoint_meta(
        args,
        dataset,
        RepairTrainingState(completed_epochs=1, global_step=7),
        interrupted=False,
    )
    assert meta["dropout"] == pytest.approx(0.1)
    assert meta["weight_decay"] == pytest.approx(1e-2)
    assert meta["augment"] is True
    assert meta["lr_scheduler"] == "cosine"
    assert meta["learning_rate"] == pytest.approx(1e-4)


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
    assert metrics.score == pytest.approx(metrics.visual_score)
    assert np.isfinite(metrics.legacy_score)
    assert np.isfinite(metrics.gradient_mae)
    assert np.isfinite(metrics.height_mae_blocks)
    assert np.isfinite(metrics.seam_mae_blocks)
    assert np.isfinite(metrics.gradient_mae_blocks)
    assert np.isfinite(metrics.laplacian_mae_blocks)
    assert np.isfinite(metrics.highpass_mae_blocks)
    assert np.isfinite(metrics.laplacian_mae)
    assert np.isfinite(metrics.highpass_mae)
    assert np.isfinite(metrics.roughness_ratio)
    assert np.isfinite(metrics.context_style_error_blocks)
    assert np.isfinite(metrics.context_roughness_ratio)
    assert np.isfinite(metrics.context_laplacian_ratio)
    assert np.isfinite(metrics.context_highpass_ratio)
    assert 0.0 <= metrics.height_within_1_block <= 1.0
    assert 0.0 <= metrics.height_within_2_blocks <= 1.0
    assert metrics.mask_pixel_count == int(mask.sum())
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
