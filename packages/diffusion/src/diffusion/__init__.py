"""Surface-oriented Minecraft terrain repair package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    'SelectionPlan',
    'TerrainDiffusionDataset',
    'TerrainRepairDataset',
    'TerrainRepairOutput',
    'TerrainRepairUNet',
    'TerrainWindowSample',
    'find_origin',
    'load_height_range',
    'plan_chunk_selection',
    'prepare_inference_inputs',
    'run_repair_job',
    'run_saved_case_jobs',
    'train_repair_step',
]

_MODULE_EXPORTS = {
    'TerrainDiffusionDataset': ('.data', 'TerrainDiffusionDataset'),
    'TerrainWindowSample': ('.data', 'TerrainWindowSample'),
    'TerrainRepairDataset': ('.repair_data', 'TerrainRepairDataset'),
    'SelectionPlan': ('.infer_inputs', 'SelectionPlan'),
    'find_origin': ('.infer_inputs', 'find_origin'),
    'load_height_range': ('.infer_inputs', 'load_height_range'),
    'plan_chunk_selection': ('.infer_inputs', 'plan_chunk_selection'),
    'prepare_inference_inputs': ('.infer_inputs', 'prepare_inference_inputs'),
    'TerrainRepairOutput': ('.repair_model', 'TerrainRepairOutput'),
    'TerrainRepairUNet': ('.repair_model', 'TerrainRepairUNet'),
    'run_repair_job': ('.repair_inference', 'run_repair_job'),
    'run_saved_case_jobs': ('.repair_inference', 'run_saved_case_jobs'),
    'train_repair_step': ('.repair_training', 'train_repair_step'),
}


def __getattr__(name: str) -> Any:
    if name not in _MODULE_EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_name, attr_name = _MODULE_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
