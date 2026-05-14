"""Diffusion package for surface-oriented Minecraft terrain repair."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    'GaussianDiffusionScheduler',
    'SelectionPlan',
    'TerrainDiffusionDataset',
    'TerrainDiffusionOutput',
    'TerrainDiffusionUNet',
    'TerrainWindowSample',
    'compute_losses',
    'find_origin',
    'load_checkpoint',
    'load_height_range',
    'multidiffusion_inpaint',
    'plan_chunk_selection',
    'prepare_inference_inputs',
    'run_inference_job',
    'save_checkpoint',
    'train_step',
]

_MODULE_EXPORTS = {
    'TerrainDiffusionDataset': ('.data', 'TerrainDiffusionDataset'),
    'TerrainWindowSample': ('.data', 'TerrainWindowSample'),
    'SelectionPlan': ('.infer_inputs', 'SelectionPlan'),
    'find_origin': ('.infer_inputs', 'find_origin'),
    'load_height_range': ('.infer_inputs', 'load_height_range'),
    'plan_chunk_selection': ('.infer_inputs', 'plan_chunk_selection'),
    'prepare_inference_inputs': ('.infer_inputs', 'prepare_inference_inputs'),
    'multidiffusion_inpaint': ('.inference', 'multidiffusion_inpaint'),
    'run_inference_job': ('.inference', 'run_inference_job'),
    'TerrainDiffusionOutput': ('.model', 'TerrainDiffusionOutput'),
    'TerrainDiffusionUNet': ('.model', 'TerrainDiffusionUNet'),
    'GaussianDiffusionScheduler': ('.scheduler', 'GaussianDiffusionScheduler'),
    'compute_losses': ('.training', 'compute_losses'),
    'load_checkpoint': ('.training', 'load_checkpoint'),
    'save_checkpoint': ('.training', 'save_checkpoint'),
    'train_step': ('.training', 'train_step'),
}


def __getattr__(name: str) -> Any:
    if name not in _MODULE_EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_name, attr_name = _MODULE_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
