import os
import json
import socket
import random
import argparse
from pathlib import Path
import sys

import numpy as np

try:
    import debugpy
except ImportError:
    debugpy = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def update_vscode_launch_file(host: str, port: int):
    """Update the .vscode/launch.json file with the new host and port."""
    launch_file_path = ".vscode/launch.json"
    # Desired configuration
    new_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "bash_debug",
                "type": "debugpy",
                "request": "attach",
                "connect": {
                    "host": host,
                    "port": port
                },
                "justMyCode": False
            },
        ]
    }

    # Ensure the .vscode directory exists
    if not os.path.exists(".vscode"):
        os.makedirs(".vscode")

    # Write the updated configuration to launch.json
    with open(launch_file_path, "w") as f:
        json.dump(new_config, f, indent=4)
    print(f"Updated {launch_file_path} with host: {host} and port: {port}")

def is_port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def setup_debug(is_main_process=True, max_retries=10, port_range=(10000, 20000)):
    if debugpy is None:
        raise ImportError("debugpy is required for setup_debug but is not installed in the active environment.")
    if is_main_process:
        host = os.environ['SLURM_NODELIST'].split(',')[0]

        for _ in range(max_retries):
            port = random.randint(*port_range)
            try:
                if is_port_in_use(host, port):
                    print(f"Port {port} is already in use, trying another...")
                    continue

                # 更新 launch.json
                update_vscode_launch_file(host, port)

                print("master_addr = ", host)
                debugpy.listen((host, port))
                print(f"Waiting for debugger attach at port {port}...", flush=True)
                debugpy.wait_for_client()
                print("Debugger attached", flush=True)
                return
            except Exception as e:
                print(f"Failed to bind to port {port}: {e}")

        raise RuntimeError("Could not find a free port for debugpy after several attempts.")


def _load_rerun():
    try:
        import rerun as rr  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "rerun is required to export .rrd files. Install the rerun-sdk package in the active environment."
        ) from exc
    return rr


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _tensor_image_to_uint8(img):
    img = _to_numpy(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


def _mask_to_uint8(mask):
    mask = _to_numpy(mask)
    return (mask > 0).astype(np.uint8) * 255


def _pointcloud_from_view(view, sample_idx):
    pts3d = _to_numpy(view["pts3d"][sample_idx])
    valid_mask = _to_numpy(view["valid_mask"][sample_idx]).astype(bool)
    rgb = _tensor_image_to_uint8(view["img"][sample_idx])

    if valid_mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    positions = pts3d[valid_mask].astype(np.float32)
    colors = rgb[valid_mask].astype(np.uint8)
    return positions, colors


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def export_dexycb_batch_rrd(
    data_root,
    output_path,
    mode="train",
    batch_size=2,
    frame_num=4,
    resolution=(224, 224),
    batch_index=0,
    sample_index=0,
    release="pi3x-dexycb-batch-debug",
    dataset_name="dexycb",
    item_id=None,
):
    from torch.utils.data import DataLoader

    from datasets.base.utils import unified_collate_fn
    from datasets.dexycb_dataset import DexYCBDataset

    rr = _load_rerun()

    dataset = DexYCBDataset(
        data_root=data_root,
        mode=mode,
        resolution=[list(resolution)],
        frame_num=frame_num,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=unified_collate_fn,
    )

    batch = None
    for current_batch_index, candidate in enumerate(loader):
        if current_batch_index == batch_index:
            batch = candidate
            break
    if batch is None:
        raise IndexError(f"batch_index={batch_index} is out of range")

    actual_batch_size = len(batch[0]["dataset"])
    if sample_index >= actual_batch_size:
        raise IndexError(
            f"sample_index={sample_index} is out of range for batch size {actual_batch_size}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init("pi3x_dexycb_batch_debug", spawn=False)
    rr.save(str(output_path))
    rr.log("world/camera", rr.ViewCoordinates.RDF)

    for frame_idx, view in enumerate(batch):
        rr.set_time_sequence("frame", frame_idx)

        rgb = _tensor_image_to_uint8(view["img"][sample_index])
        hand_mask = _mask_to_uint8(view["hand"]["mask"][sample_index])
        object_mask = _mask_to_uint8(view["object"]["mask"][sample_index])
        point_positions, point_colors = _pointcloud_from_view(view, sample_idx=sample_index)

        rr.log("frames/rgb", rr.Image(rgb))
        rr.log("frames/hand_mask", rr.Image(hand_mask))
        rr.log("frames/object_mask", rr.Image(object_mask))

        if point_positions.size == 0:
            rr.log("world/pointcloud", rr.Clear(recursive=False))
        else:
            rr.log(
                "world/pointcloud",
                rr.Points3D(
                    positions=point_positions,
                    colors=point_colors,
                ),
            )

    track_label = batch[0]["label"][sample_index]
    instances = [view["instance"][sample_index] for view in batch]
    item_id = item_id or f"{track_label}/batch{batch_index:04d}/sample{sample_index:02d}"

    metadata = {
        "data_root": str(Path(data_root).resolve()),
        "mode": mode,
        "batch_index": batch_index,
        "sample_index": sample_index,
        "batch_size": batch_size,
        "frame_num": frame_num,
        "resolution": list(resolution),
        "track_label": track_label,
        "instances": instances,
    }
    meta_path = _write_json(output_path.parent / "meta.json", metadata)
    manifest = {
        "schema_version": "1.0",
        "release": release,
        "dataset": dataset_name,
        "item_id": item_id,
        "metadata": metadata,
        "artifacts": [
            {
                "role": "interactive_rrd",
                "local_path": str(output_path.resolve()),
                "remote_name": output_path.name,
                "visibility": "private",
                "content_type": "application/octet-stream",
            },
            {
                "role": "metadata_json",
                "local_path": str(meta_path.resolve()),
                "remote_name": "meta.json",
                "visibility": "private",
                "content_type": "application/json",
            },
        ],
    }
    manifest_path = _write_json(output_path.parent / "manifest.json", manifest)
    return output_path, meta_path, manifest_path


def _build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    export_parser = subparsers.add_parser("export-dexycb-batch-rrd")
    export_parser.add_argument("--data-root", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--mode", default="train")
    export_parser.add_argument("--batch-size", type=int, default=2)
    export_parser.add_argument("--frame-num", type=int, default=4)
    export_parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
    export_parser.add_argument("--batch-index", type=int, default=0)
    export_parser.add_argument("--sample-index", type=int, default=0)
    export_parser.add_argument("--release", default="pi3x-dexycb-batch-debug")
    export_parser.add_argument("--dataset-name", default="dexycb")
    export_parser.add_argument("--item-id", default=None)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "export-dexycb-batch-rrd":
        output_path, meta_path, manifest_path = export_dexycb_batch_rrd(
            data_root=args.data_root,
            output_path=args.output,
            mode=args.mode,
            batch_size=args.batch_size,
            frame_num=args.frame_num,
            resolution=tuple(args.resolution),
            batch_index=args.batch_index,
            sample_index=args.sample_index,
            release=args.release,
            dataset_name=args.dataset_name,
            item_id=args.item_id,
        )
        print(f"Saved {output_path}")
        print(f"Saved {meta_path}")
        print(f"Saved {manifest_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
