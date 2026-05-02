"""Inspect the collated batch structure for a concrete dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.base.utils import unified_collate_fn
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS.keys()))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
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
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=unified_collate_fn,
    )
    batch = next(iter(loader))
    print(f"views_per_sample={len(batch)}")
    for view_idx, view in enumerate(batch):
        print(f"[batched view {view_idx}]")
        for key in sorted(view.keys()):
            value = view[key]
            shape = getattr(value, "shape", None)
            if shape is not None:
                print(f"  {key}: shape={tuple(shape)}")
            else:
                print(f"  {key}: type={type(value).__name__}")


if __name__ == "__main__":
    main()
