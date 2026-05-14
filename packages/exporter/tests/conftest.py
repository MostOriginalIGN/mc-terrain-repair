from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPORTER_SRC = ROOT / "packages" / "exporter" / "src"
DATASET_SRC = ROOT / "packages" / "dataset" / "src"

for src_path in (str(EXPORTER_SRC), str(DATASET_SRC)):
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
