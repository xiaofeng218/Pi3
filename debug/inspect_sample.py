"""Inspect a dataset sample without touching the trainer stack."""

from __future__ import annotations

import argparse
from pprint import pprint
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.scannet_dataset import ScannetDataset
from datasets.tartanair_dataset import TarTanAirDataset
from datasets.co3dv2_dataset import CO3DV2Dataset
from datasets.dexycb_dataset import DexYCBDataset


DATASETS = {
    "scannet": ScannetDataset,
    "tartanair": TarTanAirDataset,
    "co3dv2": CO3DV2Dataset,
    "dexycb": DexYCBDataset,
}


def describe_value(key: str, value) -> str:
    if hasattr(value, "shape"):
        dtype = getattr(value, "dtype", type(value).__name__)
        return f"{key}: shape={tuple(value.shape)} dtype={dtype}"
    return f"{key}: type={type(value).__name__} value={value}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS.keys()))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
    parser.add_argument("--frame-num", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_cls = DATASETS[args.dataset]
    dataset = dataset_cls(
        data_root=args.data_root,
        mode=args.mode,
        resolution=[args.resolution],
        frame_num=args.frame_num,
    )
    sample = dataset[args.index]
    print(f"sample_views={len(sample)}")
    for view_idx, view in enumerate(sample):
        print(f"[view {view_idx}]")
        for key in sorted(view.keys()):
            print(" ", describe_value(key, view[key]))
        print(" ", f"valid_ratio={float(np.mean(view['valid_mask'])):.6f}")
        print(" ", f"finite_pose={bool(np.isfinite(view['camera_pose']).all())}")
    pprint(getattr(dataset, "this_views_info", {}))


if __name__ == "__main__":
    main()
