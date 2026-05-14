from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diffusion.data import TerrainDiffusionDataset
from diffusion.inference import multidiffusion_inpaint
from diffusion.model import TerrainDiffusionOutput, TerrainDiffusionUNet
from diffusion.scheduler import GaussianDiffusionScheduler


class DummyModel(torch.nn.Module):
    def __init__(self, num_material_classes: int = 17):
        super().__init__()
        self.num_material_classes = num_material_classes

    def forward(self, noisy_height, known_height, mask, known_material, gradients, timesteps):
        noise_pred = torch.zeros_like(noisy_height)
        logits = torch.zeros(
            noisy_height.shape[0],
            self.num_material_classes,
            noisy_height.shape[2],
            noisy_height.shape[3],
            device=noisy_height.device,
        )
        logits[:, 3] = 1.0
        return TerrainDiffusionOutput(noise_pred=noise_pred, material_logits=logits)


def _write_fake_export(export_dir: Path, width_chunks: int = 8, height_chunks: int = 8) -> None:
    for chunk_x in range(width_chunks):
        for chunk_z in range(height_chunks):
            surface = np.full((16, 16), chunk_x * 100 + chunk_z, dtype=np.int16)
            blocks = np.zeros((16, 16, 40), dtype=np.int8)
            blocks[:, :, 32] = (chunk_x + chunk_z) % 16 + 1
            np.save(export_dir / f'surface_{chunk_x}_{chunk_z}.npy', surface)
            np.save(export_dir / f'chunk_{chunk_x}_{chunk_z}.npy', blocks)


def test_diffusion_dataset_model_and_inference(tmp_path) -> None:
    export_dir = tmp_path / 'export'
    export_dir.mkdir()
    _write_fake_export(export_dir)

    dataset = TerrainDiffusionDataset(export_dir, tile_size=128, mask_mode='rectangle', seed=7, cache_arrays=False)
    sample = dataset[0]

    assert sample['target_height'].shape == (1, 128, 128)
    assert sample['target_material'].shape == (128, 128)
    assert sample['mask'].shape == (1, 128, 128)
    assert sample['mask'].sum().item() > 0

    model = TerrainDiffusionUNet(num_material_classes=dataset.num_material_classes)
    batch = 2
    noisy_height = torch.randn(batch, 1, 128, 128)
    known_height = torch.randn(batch, 1, 128, 128)
    mask = torch.zeros(batch, 1, 128, 128)
    known_material = torch.zeros(batch, 128, 128, dtype=torch.long)
    gradients = torch.randn(batch, 2, 128, 128)
    timesteps = torch.randint(0, 1000, (batch,))

    outputs = model(noisy_height, known_height, mask, known_material, gradients, timesteps)
    assert outputs.noise_pred.shape == (batch, 1, 128, 128)
    assert outputs.material_logits.shape == (batch, dataset.num_material_classes, 128, 128)

    scheduler = GaussianDiffusionScheduler(num_train_timesteps=8).to('cpu')
    result = multidiffusion_inpaint(
        model=DummyModel(num_material_classes=dataset.num_material_classes),
        scheduler=scheduler,
        known_height=torch.zeros(1, 1, 192, 224),
        known_material=torch.zeros(1, 192, 224, dtype=torch.long),
        mask=torch.ones(1, 1, 192, 224),
        tile_size=128,
        overlap=32,
        num_steps=4,
    )

    assert result['height'].shape == (1, 1, 192, 224)
    assert result['material_logits'].shape == (1, dataset.num_material_classes, 192, 224)
    assert result['material'].shape == (1, 192, 224)
    assert torch.all(result['material'] == 3)
