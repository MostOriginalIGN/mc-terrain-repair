"""Prepare masked U-Net repair inputs from exported terrain arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
UNET_SRC = ROOT / 'packages' / 'unet' / 'src'
EXPORTER_SRC = ROOT / 'packages' / 'exporter' / 'src'
DATASET_SRC = ROOT / 'packages' / 'dataset' / 'src'

for src_path in (str(UNET_SRC), str(EXPORTER_SRC), str(DATASET_SRC)):
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from unet.infer_inputs import prepare_inference_inputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare known_height, known_material, known_support, and mask arrays for U-Net repair.')
    parser.add_argument('--export-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--checkpoint', default=None, help='Optional U-Net repair checkpoint to reuse saved height normalization range')
    parser.add_argument('--tile-size', type=int, default=128)
    parser.add_argument('--origin-chunk-x', type=int, default=None)
    parser.add_argument('--origin-chunk-z', type=int, default=None)
    parser.add_argument('--mask-top', type=int, default=48)
    parser.add_argument('--mask-left', type=int, default=48)
    parser.add_argument('--mask-height', type=int, default=32)
    parser.add_argument('--mask-width', type=int, default=32)
    args = parser.parse_args()

    metadata = prepare_inference_inputs(
        export_dir=args.export_dir,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        tile_size=args.tile_size,
        origin_chunk_x=args.origin_chunk_x,
        origin_chunk_z=args.origin_chunk_z,
        mask_top=args.mask_top,
        mask_left=args.mask_left,
        mask_height=args.mask_height,
        mask_width=args.mask_width,
    )
    print(f"Prepared inference inputs in {Path(args.out_dir).expanduser().resolve()} for origin ({metadata['origin_chunk_x']}, {metadata['origin_chunk_z']})")


if __name__ == '__main__':
    main()
