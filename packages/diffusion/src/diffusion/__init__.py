"""Diffusion package for surface-oriented Minecraft terrain repair."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GaussianDiffusionScheduler",
    "TerrainDiffusionDataset",
    "TerrainDiffusionOutput",
    "TerrainDiffusionUNet",
    "TerrainWindowSample",
    "compute_losses",
    "load_checkpoint",
    "multidiffusion_inpaint",
    "save_checkpoint",
    "train_step",
]

_MODULE_EXPORTS = {
    "TerrainDiffusionDataset": (".data", "TerrainDiffusionDataset"),
    "TerrainWindowSample": (".data", "TerrainWindowSample"),
    "multidiffusion_inpaint": (".inference", "multidiffusion_inpaint"),
    "TerrainDiffusionOutput": (".model", "TerrainDiffusionOutput"),
    "TerrainDiffusionUNet": (".model", "TerrainDiffusionUNet"),
    "GaussianDiffusionScheduler": (".scheduler", "GaussianDiffusionScheduler"),
    "compute_losses": (".training", "compute_losses"),
    "load_checkpoint": (".training", "load_checkpoint"),
    "save_checkpoint": (".training", "save_checkpoint"),
    "train_step": (".training", "train_step"),
}


def __getattr__(name: str) -> Any:
    if name not in _MODULE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _MODULE_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
