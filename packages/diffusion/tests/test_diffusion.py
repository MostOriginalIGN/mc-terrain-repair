from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diffusion.data import TerrainDiffusionDataset
from diffusion.infer_inputs import plan_chunk_selection, prepare_inference_inputs
from diffusion.inference import multidiffusion_inpaint, run_inference_job
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




def _write_dummy_checkpoint(path: Path, num_material_classes: int) -> None:
    checkpoint = {
        'model_state': TerrainDiffusionUNet(num_material_classes=num_material_classes).state_dict(),
        'optimizer_state': None,
        'scheduler_config': GaussianDiffusionScheduler().config.__dict__,
        'num_material_classes': num_material_classes,
        'meta': {'height_min': 10.0, 'height_max': 50.0},
    }
    torch.save(checkpoint, path)

def _write_fake_export(export_dir: Path, width_chunks: int = 8, height_chunks: int = 8) -> None:
    for chunk_x in range(width_chunks):
        for chunk_z in range(height_chunks):
            surface = np.full((16, 16), chunk_x * 100 + chunk_z, dtype=np.int16)
            blocks = np.zeros((16, 16, 40), dtype=np.int8)
            blocks[:, :, 32] = (chunk_x + chunk_z) % 16 + 1
            np.save(export_dir / f'surface_{chunk_x}_{chunk_z}.npy', surface)
            np.save(export_dir / f'chunk_{chunk_x}_{chunk_z}.npy', blocks)


def test_diffusion_dataset_model_inference_and_input_prep(tmp_path) -> None:
    export_dir = tmp_path / 'export'
    export_dir.mkdir()
    _write_fake_export(export_dir)

    dataset = TerrainDiffusionDataset(export_dir, tile_size=128, mask_mode='rectangle', seed=7, cache_arrays=False)
    sample = dataset[0]

    assert sample['target_height'].shape == (1, 128, 128)
    assert sample['target_material'].shape == (128, 128)
    assert sample['mask'].shape == (1, 128, 128)
    assert sample['mask'].sum().item() > 0

    planner_dataset = TerrainDiffusionDataset(export_dir, tile_size=128, mask_mode='none', cache_arrays=False)
    plan = plan_chunk_selection(
        planner_dataset.window_origins,
        planner_dataset.chunks_per_side,
        selected_min_chunk_x=2,
        selected_min_chunk_z=3,
        selected_max_chunk_x=3,
        selected_max_chunk_z=4,
    )
    assert plan.origin_chunk_x == 0
    assert plan.origin_chunk_z == 0
    assert plan.mask_left == 32
    assert plan.mask_top == 48

    inputs_dir = tmp_path / 'inputs'
    metadata = prepare_inference_inputs(
        export_dir=export_dir,
        out_dir=inputs_dir,
        tile_size=128,
        origin_chunk_x=plan.origin_chunk_x,
        origin_chunk_z=plan.origin_chunk_z,
        mask_top=plan.mask_top,
        mask_left=plan.mask_left,
        mask_height=plan.mask_height,
        mask_width=plan.mask_width,
    )
    mask = np.load(inputs_dir / 'mask.npy')
    assert metadata['origin_chunk_x'] == 0
    assert metadata['origin_chunk_z'] == 0
    assert mask.shape == (128, 128)
    assert float(mask.sum()) == float(plan.mask_height * plan.mask_width)

    model = TerrainDiffusionUNet(num_material_classes=dataset.num_material_classes)
    batch = 2
    noisy_height = torch.randn(batch, 1, 128, 128)
    known_height = torch.randn(batch, 1, 128, 128)
    mask_tensor = torch.zeros(batch, 1, 128, 128)
    known_material = torch.zeros(batch, 128, 128, dtype=torch.long)
    gradients = torch.randn(batch, 2, 128, 128)
    timesteps = torch.randint(0, 1000, (batch,))

    outputs = model(noisy_height, known_height, mask_tensor, known_material, gradients, timesteps)
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

    checkpoint = tmp_path / 'diffusion.pt'
    _write_dummy_checkpoint(checkpoint, dataset.num_material_classes)
    outputs = run_inference_job(
        checkpoint=checkpoint,
        known_height_path=inputs_dir / 'known_height.npy',
        known_material_path=inputs_dir / 'known_material.npy',
        mask_path=inputs_dir / 'mask.npy',
        out_dir=tmp_path / 'outputs',
        tile_size=128,
        overlap=32,
        num_steps=2,
    )
    assert outputs['preview'].exists()
    assert outputs['material_preview'].exists()
    assert outputs['preview_panel'].exists()
