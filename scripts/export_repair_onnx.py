"""Export a trained terrain repair checkpoint to ONNX for Java inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
UNET_SRC = ROOT / "packages" / "unet" / "src"
if str(UNET_SRC) not in sys.path:
    sys.path.insert(0, str(UNET_SRC))

from unet.repair_onnx import export_repair_onnx  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export deterministic terrain repair model to ONNX (+ JSON metadata for plugin use)."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to repair.pt checkpoint.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .onnx path (default: <checkpoint_stem>.onnx beside checkpoint).",
    )
    parser.add_argument("--tile-size", type=int, default=128, help="Spatial size used for tracing.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip onnx/onnxruntime checker after export.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else checkpoint.with_suffix(".onnx")
    )

    exported = export_repair_onnx(
        checkpoint=checkpoint,
        output_path=output,
        tile_size=args.tile_size,
        opset_version=args.opset,
        verify=not args.no_verify,
    )
    metadata_path = exported.with_suffix(".json")
    print(f"Exported ONNX model to {exported}")
    print(f"Wrote plugin metadata to {metadata_path}")


if __name__ == "__main__":
    main()
