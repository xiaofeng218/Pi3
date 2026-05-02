import torch
import argparse
import numpy as np
import os
from contextlib import nullcontext
from pi3.utils.basic import load_multimodal_data
from pi3.utils.geometry import depth_edge, recover_intrinsic_from_rays_d
from pi3.utils.vis_export import (
    build_manifest,
    collect_masked_points_and_colors,
    default_item_id,
    plan_sequence_output,
    write_manifest,
    write_rrd_sequence,
)
from pi3.models.pi3x import Pi3X

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    
    # parser.add_argument("--conditions_path", type=str, default='examples/room/condition.npz',
    parser.add_argument("--conditions_path", type=str, default=None,
                        help="Optional path to a .npz file containing 'poses', 'depths', 'intrinsics'.")

    parser.add_argument("--output_root", type=str, default='output',
                        help="Root directory for exported artifacts.")
    parser.add_argument("--job_name", type=str, default='pi3_mm',
                        help="Subdirectory under output_root used for this export job.")
    parser.add_argument("--item_id", type=str, default=None,
                        help="Optional vis_upload item_id. Defaults to the input stem or directory name.")
    parser.add_argument("--release", type=str, default='pi3-mm',
                        help="Release name written into manifest.json.")
    parser.add_argument("--dataset", type=str, default='pi3',
                        help="Dataset name written into manifest.json.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # 1. Prepare input data
    device = torch.device(args.device)

    # Load optional conditions from .npz
    poses = None
    depths = None
    intrinsics = None

    if args.conditions_path is not None and os.path.exists(args.conditions_path):
        print(f"Loading conditions from {args.conditions_path}...")
        data_npz = np.load(args.conditions_path, allow_pickle=True)

        poses = data_npz['poses']             # Expected (N, 4, 4) OpenCV camera-to-world
        depths = data_npz['depths']           # Expected (N, H, W)
        intrinsics = data_npz['intrinsics']   # Expected (N, 3, 3)

    conditions = dict(
        intrinsics=intrinsics,
        poses=poses,
        depths=depths
    )

    # Load images (Required)
    imgs, conditions = load_multimodal_data(args.data_path, conditions, interval=args.interval, device=device) 
    use_multimodal = any(v is not None for v in conditions.values())
    if not use_multimodal:
        print("No multimodal conditions found. Disable multimodal branch to reduce memory usage.")

    # 2. Prepare model
    print(f"Loading model...")
    if args.ckpt is not None:
        model = Pi3X(use_multimodal=use_multimodal).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight, strict=False)
    else:
        try:
            model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        except Exception as exc:
            print(f"Failed to load pretrained Pi3X weights: {exc}")
            print("Falling back to randomly initialized weights. Provide --ckpt for meaningful predictions.")
            model = Pi3X(use_multimodal=use_multimodal).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`
        if not use_multimodal:
            model.disable_multimodal()
    model = model.to(device)

    """
    Args:
        imgs (torch.Tensor): Input RGB images valued in [0, 1].
            Shape: (B, N, 3, H, W).
        intrinsics (torch.Tensor, optional): Camera intrinsic matrices.
            Shape: (B, N, 3, 3).
            Values are in pixel coordinates (not normalized).
        rays (torch.Tensor, optional): Pre-computed ray directions (unit vectors).
            Shape: (B, N, H, W, 3).
            Can replace `intrinsics` as a geometric condition.
        poses (torch.Tensor, optional): Camera-to-World matrices.
            Shape: (B, N, 4, 4).
            Coordinate system: OpenCV convention (Right-Down-Forward).
        depths (torch.Tensor, optional): Ground truth or prior depth maps.
            Shape: (B, N, H, W).
            Invalid values (e.g., sky or missing data) should be set to 0.
        mask_add_depth (torch.Tensor, optional): Mask for depth condition.
            Shape: (B, N, N).
        mask_add_ray (torch.Tensor, optional): Mask for ray/intrinsic condition.
            Shape: (B, N, N).
        mask_add_pose (torch.Tensor, optional): Mask for pose condition.
            Shape: (B, N, N).
            Note: Requires at least two frames to be True to establish a meaningful
            coordinate system (absolute pose for a single frame provides no relative constraint).
    """

    # 3. Infer
    print("Running model inference...")
    use_cuda = device.type == 'cuda'
    if use_cuda:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        autocast_ctx = torch.amp.autocast('cuda', dtype=dtype)
    else:
        autocast_ctx = nullcontext()
    
    with torch.no_grad():
        with autocast_ctx:
            res = model(
                imgs=imgs, 
                **conditions
            )

    # 3.5 Recover intrinsic from rays_d
    rays_d = torch.nn.functional.normalize(res['local_points'], dim=-1)
    K = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
    print(f"Recovered frist frame intrinsic: \n{K[0, 0].cpu().numpy()}")
    if conditions['intrinsics'] is not None:
        print(f"Original frist frame intrinsic: \n{conditions['intrinsics'][0, 0].cpu().numpy()}")

    # 4. process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Save points as one frame-wise Rerun sequence plus vis_upload manifest
    item_id = args.item_id or default_item_id(args.data_path)
    export_plan = plan_sequence_output(
        output_root=args.output_root,
        job_name=args.job_name,
        item_id=item_id,
    )
    print(f"Saving Rerun sequence to: {export_plan.rrd_path}")

    frame_points, frame_colors = collect_masked_points_and_colors(
        res['points'][0].detach().cpu().numpy(),
        imgs[0].permute(0, 2, 3, 1).detach().cpu().numpy(),
        masks.detach().cpu().numpy(),
    )
    write_rrd_sequence(export_plan.rrd_path, frame_points, frame_colors)

    manifest = build_manifest(
        item_id=export_plan.item_id,
        rrd_path=export_plan.rrd_path,
        release=args.release,
        dataset=args.dataset,
        metadata={
            "source_path": args.data_path,
            "conditions_path": args.conditions_path,
            "interval": args.interval,
            "num_frames": len(frame_points),
        },
    )
    write_manifest(export_plan.manifest_path, manifest)
    print(f"Saved manifest to: {export_plan.manifest_path}")
    print("Done.")
