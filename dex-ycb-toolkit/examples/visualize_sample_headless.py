"""Save RGB, segmentation, joints, and overlay visualizations for one sample."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dex_ycb_toolkit.headless_vis import load_sample_visuals, save_visualizations


def parse_args():
    parser = argparse.ArgumentParser(description="Headless DexYCB sample visualization.")
    parser.add_argument("--color_file", required=True, help="Path to color_XXXXXX.jpg")
    parser.add_argument("--label_file", required=True, help="Path to labels_XXXXXX.npz")
    parser.add_argument(
        "--output_dir",
        default="output/headless_vis",
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional filename prefix. Defaults to the color image stem.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    color_path = Path(args.color_file)
    prefix = args.prefix or color_path.stem

    images = load_sample_visuals(args.color_file, args.label_file)
    saved = save_visualizations(images, args.output_dir, prefix)

    for name, path in saved.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
