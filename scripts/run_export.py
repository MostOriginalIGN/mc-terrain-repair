"""CLI entrypoint for exporting Minecraft terrain chunks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPORTER_SRC = ROOT / 'packages' / 'exporter' / 'src'
DATASET_SRC = ROOT / 'packages' / 'dataset' / 'src'

for src_path in (str(EXPORTER_SRC), str(DATASET_SRC)):
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from exporter.export import export_chunks  # noqa: E402

OVERWORLD_RELATIVE_PATH = Path('dimensions/minecraft/overworld')


def resolve_world_path(path_str: str) -> Path:
    base_path = Path(path_str).expanduser().resolve()
    if not base_path.exists():
        raise SystemExit(f'World path does not exist: {base_path}')

    candidates = [
        base_path,
        base_path / OVERWORLD_RELATIVE_PATH,
    ]
    for candidate in candidates:
        if (candidate / 'region').is_dir():
            return candidate

    raise SystemExit(
        'World path must be either an overworld directory containing region/ '
        f'or a save root containing {OVERWORLD_RELATIVE_PATH}/region: {base_path}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Export a surface-anchored terrain dataset from a Minecraft world.')
    parser.add_argument(
        '--world',
        required=True,
        help='Path to a Minecraft save root or directly to the overworld directory',
    )
    parser.add_argument('--out', required=True, help='Output directory for exported `.npy` files')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of chunks to export')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Deterministic shuffle inside region-local center-first batches',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        metavar='N',
        help='Parallel worker processes (default: 1; increase only after benchmarking your machine)',
    )
    args = parser.parse_args()

    world_path = resolve_world_path(args.world)
    export_chunks(str(world_path), args.out, limit=args.limit, seed=args.seed, workers=args.workers)
    print(f'Export completed for world {world_path} into {Path(args.out).resolve()}')


if __name__ == '__main__':
    main()
