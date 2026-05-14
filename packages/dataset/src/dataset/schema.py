"""Shared naming helpers for exported chunk artifacts."""

from __future__ import annotations

from pathlib import Path

MANIFEST_FILENAME = "manifest.json"


def chunk_blocks_filename(chunk_x: int, chunk_z: int) -> str:
    """Return the block slab file name for one chunk."""
    return f"chunk_{chunk_x}_{chunk_z}.npy"


def surface_filename(chunk_x: int, chunk_z: int) -> str:
    """Return the surface height file name for one chunk."""
    return f"surface_{chunk_x}_{chunk_z}.npy"


def manifest_path(output_dir: str | Path) -> Path:
    """Return the manifest path for an export directory."""
    return Path(output_dir) / MANIFEST_FILENAME
