"""Basic import smoke test for the root data pipeline."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODULES = [
    "datasets",
    "datasets.base.base_dataset",
    "datasets.base.batched_sampler",
    "datasets.base.transforms",
    "datasets.scannet_dataset",
    "datasets.tartanair_dataset",
    "datasets.co3dv2_dataset",
    "datasets.dexycb_dataset",
]


def main() -> None:
    for module_name in MODULES:
        importlib.import_module(module_name)
        print(f"OK {module_name}")


if __name__ == "__main__":
    main()
