from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diffusion.data import TerrainDiffusionDataset
from diffusion.model import TerrainDiffusionUNet
from diffusion.scheduler import GaussianDiffusionScheduler
from diffusion.training import (
    TrainingState,
    build_checkpoint_meta,
    compute_losses,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
    train_step,
)


def _write_fake_export(export_dir: Path, width_chunks: int = 8, height_chunks: int = 8) -> None:
    for chunk_x in range(width_chunks):
        for chunk_z in range(height_chunks):
            surface = np.full((16, 16), 64 + chunk_x + chunk_z, dtype=np.int16)
            blocks = np.zeros((16, 16, 40), dtype=np.int8)
            blocks[:, :, 32] = (chunk_x * height_chunks + chunk_z) % 16
            np.save(export_dir / f'surface_{chunk_x}_{chunk_z}.npy', surface)
            np.save(export_dir / f'chunk_{chunk_x}_{chunk_z}.npy', blocks)


def test_trainer_step_and_checkpoint_roundtrip(tmp_path) -> None:
    export_dir = tmp_path / 'export'
    export_dir.mkdir()
    _write_fake_export(export_dir)

    dataset = TerrainDiffusionDataset(export_dir, tile_size=128, mask_mode='rectangle', seed=11)
    sample = dataset[0]
    batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) and value.ndim in (1, 2, 3) else value for key, value in sample.items()}

    model = TerrainDiffusionUNet(num_material_classes=dataset.num_material_classes)
    scheduler = GaussianDiffusionScheduler(num_train_timesteps=16).to('cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = compute_losses(model, scheduler, batch)
    assert losses.total_loss.item() > 0

    train_step(model, optimizer, scheduler, batch)

    checkpoint = tmp_path / 'diffusion.pt'
    args = type('Args', (), {
        'tile_size': 128,
        'stride_chunks': 1,
        'export_dir': str(export_dir),
    })()
    meta = build_checkpoint_meta(args, dataset, TrainingState(completed_epochs=3, global_step=17), interrupted=False)
    save_checkpoint(checkpoint, model, optimizer, scheduler, meta=meta)

    reloaded = TerrainDiffusionUNet(num_material_classes=dataset.num_material_classes)
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-4)
    payload = load_checkpoint(checkpoint, reloaded, optimizer=reloaded_optimizer)
    state = restore_training_state(payload)

    assert payload['meta']['tile_size'] == 128
    assert payload['meta']['epoch'] == 3
    assert payload['meta']['global_step'] == 17
    assert payload['meta']['interrupted'] is False
    assert state.completed_epochs == 3
    assert state.global_step == 17
    assert reloaded_optimizer.state_dict()['state']
