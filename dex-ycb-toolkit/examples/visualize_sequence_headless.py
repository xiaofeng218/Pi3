"""Save visualization videos for one DexYCB camera directory."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dex_ycb_toolkit.headless_vis import save_sequence_videos


def parse_args():
    parser = argparse.ArgumentParser(description="Headless DexYCB sequence visualization.")
    parser.add_argument(
        "--camera_dir",
        required=True,
        help="Path to a camera directory containing color_*.jpg and labels_*.npz",
    )
    parser.add_argument(
        "--output_dir",
        default="output/headless_vis_sequence",
        help="Directory to save visualization videos.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional filename prefix. Defaults to the camera directory name.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for output videos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    saved = save_sequence_videos(
        camera_dir=args.camera_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
        fps=args.fps,
    )
    for name, path in saved.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
