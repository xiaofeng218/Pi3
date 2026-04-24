import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from copy import deepcopy
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin

from .layers.conv_head import ConvHead
from .layers.camera_head import CameraHead
from .layers.hand_token_adapter import HandTokenAdapter
from .layers.object_pose_head import ObjectPoseHead
from .layers.object_query_adapter import ObjectQueryAdapter
from .dinov2.layers import Mlp, PatchEmbed
from .layers.attention import FlashAttentionRope
from .layers.block import BlockRope, PoseInjectBlock
from .layers.pos_embed import RoPE2D, PositionGetter
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from ..utils.geometry import se3_inverse, get_pixel, homogenize_points
from .layers.transformer_head import TransformerDecoder, ContextOnlyTransformerDecoder
from .layers.dual_stream_routing import flatten_object_global_memory
from .hamer.config import get_config as get_hamer_config
from .hamer.hand_mano_head import HandMANOHead


class Pi3X(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            ckpt=None,    
            use_multimodal=True,
            hamer_config_file=None,
            hamer_cache_dir=None,
        ):
        super().__init__()

        self.use_multimodal = use_multimodal
        self.hamer_config_file = hamer_config_file
        self.hamer_cache_dir = hamer_cache_dir

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        freq = 100
        self.rope = RoPE2D(freq=freq)
        self.position_getter = PositionGetter()

        # ----------------------
        #        Decoder
        # ----------------------
        dec_embed_dim = 1024
        dec_num_heads = 16
        mlp_ratio = 4
        dec_depth = 36      
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
        ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # -----------------------
        #       multi-modal
        # -----------------------
        if use_multimodal:
            ## Depth encoder
            self.depth_encoder = deepcopy(self.encoder)
            del self.depth_encoder.patch_embed
            self.depth_encoder.patch_embed = PatchEmbed(img_size=224, patch_size=14, in_chans=2, embed_dim=1024)
            self.depth_emb = nn.Parameter(torch.zeros(1, 1, 1024))

            ## Ray embedding
            self.ray_embed = PatchEmbed(img_size=224, patch_size=14, in_chans=2, embed_dim=1024)
            nn.init.constant_(self.ray_embed.proj.weight, 0)
            nn.init.constant_(self.ray_embed.proj.bias, 0)

            ## Pose inject blocks
            self.pose_inject_blk = nn.ModuleList([PoseInjectBlock(
                dim=1024,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
            ) for _ in range(5)])


        # ------------------------------
        #           Head
        # ------------------------------
        ## --------------- Point ---------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=1024,
            rope=self.rope,
        )
        # self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
        self.point_head = ConvHead(
                num_features=4, 
                dim_in=dec_embed_dim,
                # projects=nn.Linear(1024, 1024),
                projects=nn.Identity(),
                dim_out=[2, 1], 
                dim_proj=1024,
                dim_upsample=[256, 128, 64],
                dim_times_res_block_hidden=2,
                num_res_blocks=2,
                res_block_norm='group_norm',
                last_res_blocks=0,
                last_conv_channels=32,
                last_conv_size=1,
                using_uv=True
            )

        ## --------------- Camera ---------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
        )
        self.camera_head = CameraHead(dim=512)

        ## --------------- Metric ---------------
        self.metric_token = nn.Parameter(torch.randn(1, 1, 2*self.dec_embed_dim))
        self.metric_decoder = ContextOnlyTransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=512,
            dec_num_heads=8,                # 8
            out_dim=512,
            rope=self.rope,
        )
        self.metric_head = nn.Linear(512, 1)
        nn.init.normal_(self.metric_token, std=1e-6)


        ## -------------- Conf ------------------
        self.conf_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=1024,
            rope=self.rope,
        )
        self.conf_head = ConvHead(
            num_features=4, 
            dim_in=dec_embed_dim,
            # projects=nn.Linear(1024, 1024),
            projects=nn.Identity(),
            dim_out=[1], 
            dim_proj=1024,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm='group_norm',
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1,
            using_uv=True
        )

        ## ------------- Hand -------------------
        self.hand_token_adapter = HandTokenAdapter(token_dim=self.dec_embed_dim, patch_size=self.patch_size)
        self.object_query_adapter = ObjectQueryAdapter(token_dim=self.dec_embed_dim, patch_size=self.patch_size)
        self.object_pose_head = ObjectPoseHead(in_dim=2 * self.dec_embed_dim, hidden_dim=self.dec_embed_dim)
        self.hand_mano_head = None

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)


    def disable_multimodal(self, free_cuda_cache: bool = True):
        """
        Disables multimodal branches and releases their modules/parameters.
        Use this when no multimodal conditions are provided.
        """
        self.use_multimodal = False
        for attr in ("depth_encoder", "depth_emb", "ray_embed", "pose_inject_blk"):
            if hasattr(self, attr):
                delattr(self, attr)

        if free_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _default_hamer_paths(self):
        repo_root = Path(__file__).resolve().parents[2]
        config_file = repo_root / "third_party" / "hamer" / "_DATA" / "hamer_ckpts" / "model_config.yaml"
        cache_dir = repo_root / "third_party" / "hamer" / "_DATA"
        return config_file, cache_dir

    def _get_hand_mano_head(self, device: torch.device):
        if self.hand_mano_head is None:
            default_config_file, default_cache_dir = self._default_hamer_paths()
            config_file = self.hamer_config_file or str(default_config_file)
            cache_dir = self.hamer_cache_dir or str(default_cache_dir)
            hamer_cfg = get_hamer_config(config_file, merge=True, cache_dir=cache_dir, update_cachedir=True)
            self.hand_mano_head = HandMANOHead(hamer_cfg, in_dim=2 * self.dec_embed_dim, hidden_dim=self.dec_embed_dim)
            self.hand_mano_head = self.hand_mano_head.to(device)
        return self.hand_mano_head

    def _object_query_reads_object_memory(self, object_query: torch.Tensor, object_memory: torch.Tensor) -> torch.Tensor:
        scale = object_query.shape[-1] ** -0.5
        attn = torch.softmax(torch.einsum("bnc,bmc->bnm", object_query, object_memory) * scale, dim=-1)
        return torch.einsum("bnm,bmc->bnc", attn, object_memory)
            
    def forward(
        self,
        imgs,
        depths=None,
        intrinsics=None,
        rays=None,
        poses=None,
        with_prior=True,
        mask_add_depth=None,
        mask_add_ray=None,
        mask_add_pose=None,
        hand_queries=None,
        hand_masks=None,
        hand_owner_index=None,
        hand_is_right=None,
        object_multiview=None,
    ):
        """
        Forward pass with optional multimodal conditions.

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

        Returns:
            dict: Model outputs containing 'points', 'conf', etc.
        """
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # encode
        rgb_patch_tokens, hidden, poses_, use_depth_mask, use_pose_mask, norm_factor = self.encode(
            imgs, 
            with_prior=with_prior, 
            depths=depths, 
            intrinsics=intrinsics, 
            poses=poses, 
            rays=rays,
            mask_add_depth=mask_add_depth,
            mask_add_ray=mask_add_ray,
            mask_add_pose=mask_add_pose,
        )
        rgb_patch_tokens = rgb_patch_tokens.reshape(B, N, -1, self.dec_embed_dim)
        hidden = hidden.reshape(B, N, -1, self.dec_embed_dim)

        hand_adapter_outputs = None
        num_hand_tokens = 0
        if hand_queries is not None:
            if hand_masks is None or hand_owner_index is None or hand_is_right is None:
                raise ValueError("hand_masks, hand_owner_index, and hand_is_right are required when hand_queries is provided")
            hand_adapter_outputs = self.hand_token_adapter(
                rgb_patch_tokens,
                hand_queries,
                hand_masks,
                hand_owner_index,
                hand_is_right,
                image_hw=(H, W),
            )
            num_hand_tokens = hand_adapter_outputs["num_hand_tokens"]

        object_adapter_outputs = None
        object_memory = None
        num_object_tokens = 0
        if object_multiview is not None:
            required = ["img", "depthmap", "camera_intrinsics", "camera_pose", "grasped_object_mask", "grasped_object_valid"]
            missing = [key for key in required if key not in object_multiview]
            if missing:
                raise ValueError(f"object_multiview missing keys: {missing}")

            object_adapter_outputs = self.object_query_adapter(
                rgb_patch_tokens,
                object_multiview["grasped_object_mask"],
                object_multiview["grasped_object_valid"],
                image_hw=(H, W),
            )
            num_object_tokens = object_adapter_outputs["object_query"].shape[2]

            obj_imgs = object_multiview["img"]
            obj_B, obj_N, _, obj_H, obj_W = obj_imgs.shape
            obj_depths = object_multiview["depthmap"] if self.use_multimodal else None
            obj_intrinsics = object_multiview["camera_intrinsics"] if self.use_multimodal else None
            obj_poses = object_multiview["camera_pose"]
            obj_rays = object_multiview.get("rays", None)
            _, obj_hidden, obj_poses_, _, obj_use_pose_mask, _ = self.encode(
                obj_imgs,
                with_prior=with_prior,
                depths=obj_depths,
                intrinsics=obj_intrinsics,
                poses=obj_poses,
                rays=obj_rays,
            )
            obj_hidden = obj_hidden.reshape(obj_B, obj_N, -1, self.dec_embed_dim)
            obj_use_pose_mask = torch.ones((obj_B, obj_N), dtype=torch.bool, device=obj_imgs.device)
            obj_hidden, _ = self.decode(
                obj_hidden,
                obj_N,
                obj_H,
                obj_W,
                obj_poses_,
                obj_use_pose_mask,
            )
            obj_hidden = obj_hidden.reshape(obj_B, obj_N, -1, 2 * self.dec_embed_dim)[..., self.dec_embed_dim:]
            object_memory = flatten_object_global_memory(obj_hidden, self.patch_start_idx)

        # decode
        hidden, pos = self.decode(
            hidden,
            N,
            H,
            W,
            poses_,
            use_pose_mask,
            hand_tokens=None if hand_adapter_outputs is None else hand_adapter_outputs["dense_tokens"],
            hand_pos=None if hand_adapter_outputs is None else hand_adapter_outputs["dense_pos"],
            object_tokens=None if object_adapter_outputs is None else object_adapter_outputs["object_query"],
            object_pos=None if object_adapter_outputs is None else object_adapter_outputs["object_query_pos"],
        )

        # # head
        scene_total_tokens = self.patch_start_idx + patch_h * patch_w + num_hand_tokens
        scene_hidden = hidden[:, :scene_total_tokens, :]
        scene_pos = pos[:, :scene_total_tokens, :]
        outputs = self.forward_head(scene_hidden, scene_pos, B, N, H, W, patch_h, patch_w, num_hand_tokens=num_hand_tokens)

        if hand_adapter_outputs is not None and num_hand_tokens > 0:
            total_seq = self.patch_start_idx + patch_h * patch_w + num_hand_tokens
            if num_object_tokens > 0:
                total_seq += num_object_tokens
            decoded = hidden.reshape(B, N, total_seq, -1)
            hand_slice = slice(self.patch_start_idx + patch_h * patch_w, self.patch_start_idx + patch_h * patch_w + num_hand_tokens)
            dense_hand_features = decoded[:, :, hand_slice, :]
            sparse_hand_features = []
            for idx in range(hand_owner_index.shape[0]):
                b_idx, n_idx, m_idx = hand_owner_index[idx].tolist()
                sparse_hand_features.append(dense_hand_features[b_idx, n_idx, m_idx])
            if sparse_hand_features:
                sparse_hand_features = torch.stack(sparse_hand_features, dim=0)
                hand_mano_head = self._get_hand_mano_head(hidden.device)
                hand_outputs = hand_mano_head(sparse_hand_features)
                outputs["pred_hand_mano_params"] = hand_outputs["pred_mano_params"]
                outputs["pred_hand_cam"] = hand_outputs["pred_cam"]
                outputs["hand_owner_index"] = hand_owner_index
                outputs["hand_is_right"] = hand_is_right
                outputs["hand_token_features"] = sparse_hand_features

        if object_adapter_outputs is not None and num_object_tokens > 0:
            total_seq = self.patch_start_idx + patch_h * patch_w + num_hand_tokens + num_object_tokens
            decoded = hidden.reshape(B, N, total_seq, -1)
            object_start = self.patch_start_idx + patch_h * patch_w + num_hand_tokens
            object_slice = slice(object_start, object_start + num_object_tokens)
            object_query_feat = decoded[:, :, object_slice, :]
            object_query_penultimate = object_query_feat[..., :self.dec_embed_dim]
            object_query_final = object_query_feat[..., self.dec_embed_dim:]

            if object_memory is not None:
                valid_mask = object_adapter_outputs["object_valid"].unsqueeze(-1).unsqueeze(-1)
                updated_object_query_final = self._object_query_reads_object_memory(
                    object_query_final.reshape(B, N * num_object_tokens, self.dec_embed_dim),
                    object_memory,
                ).reshape(B, N, num_object_tokens, self.dec_embed_dim)
                object_query_final = torch.where(valid_mask, updated_object_query_final, object_query_final)

            object_query_feat = torch.cat([object_query_penultimate, object_query_final], dim=-1)
            object_pose = self.object_pose_head(object_query_feat.reshape(B, N * num_object_tokens, -1))
            outputs["pred_object_rot6d"] = object_pose["rot6d"].reshape(B, N, num_object_tokens, 6).squeeze(2)
            outputs["pred_object_trans"] = object_pose["trans"].reshape(B, N, num_object_tokens, 3).squeeze(2)
            outputs["pred_object_log_scale"] = object_pose["log_scale"].reshape(B, N, num_object_tokens, 1).squeeze(2)
            outputs["pred_object_scale"] = object_pose["scale"].reshape(B, N, num_object_tokens, 1).squeeze(2)
            outputs["object_valid"] = object_adapter_outputs["object_valid"]

        return outputs
    
    def encode(
        self, 
        imgs, 
        with_prior=True,
        depths=None,
        rays=None,
        intrinsics=None,
        poses=None,
        mask_add_depth=None,
        mask_add_ray=None,
        mask_add_pose=None,
    ):
        B, N, _, H, W = imgs.shape
        device = imgs.device

        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        rgb_patch_tokens = self.encoder(imgs, is_training=True)["x_norm_patchtokens"]
        hidden = rgb_patch_tokens

        if self.use_multimodal:
            with torch.amp.autocast(device_type='cuda', enabled=False):
                if with_prior is True:
                    p_depth = p_ray = p_pose = 1.0
                else:
                    p_depth = p_ray = p_pose = 0.0

                if depths is None:
                    p_depth = 0.0
                    depths = torch.zeros((B, N, H, W), device=imgs.device)

                if rays is not None:
                    rays = rays[..., :2] / (rays[..., 2:3] + 1e-6)
                else:
                    if intrinsics is None:
                        p_ray = 0.0
                        rays = torch.zeros((B, N, H, W, 2), device=imgs.device)
                    else:
                        pix = torch.from_numpy(get_pixel(H, W).T.reshape(H, W, 3)).to(device).float()[None].repeat(B, 1, 1, 1)
                        rays = torch.einsum('bnij, bhwj -> bnhwi', torch.inverse(intrinsics), pix)[..., :2]
                        # rays = F.normalize(rays, dim=-1).reshape(B, N, H, W, 3)                   # don't normalize, so the pred['xy'] is the same as input rays

                if poses is None:
                    p_pose = 0.0
                    poses = torch.eye(4, device=device)[None, None].repeat(B, N, 1, 1)
                else:
                    assert rays is not None                     # rays should be along with poses
                    
                if mask_add_depth is None:
                    mask_add_depth = torch.rand((B, N), device=device) <= p_depth
                if mask_add_ray is None:
                    mask_add_ray = torch.rand((B, N), device=device) <= p_ray
                if mask_add_pose is None:
                    mask_add_pose = torch.rand((B, N), device=device) <= p_pose

                # pose is injected relatively. so at least two frame should be true.
                num_valid_pose = mask_add_pose.sum(dim=1)
                bad_indices = (num_valid_pose == 1)
                mask_add_pose[bad_indices] = False

                # normalize depth and pose
                normalized_depths, dep_median = self.normalize_depth(depths, method='mean')
                scale_aug = 0.8 + torch.rand((B,), device=device) * 0.4
                normalized_depths /= scale_aug.view(B, 1, 1, 1)
                dep_median *= scale_aug

                depths_masks = (normalized_depths > 0).float()
                depths_masks = depths_masks.reshape(B*N, 1, H, W)

                poses_ = torch.einsum('bij, bnjk -> bnik', se3_inverse(poses[:, 0]), poses)
                poses_[..., :3, 3] /= dep_median.view(B, 1, 1)

                # noramlize for the batch not using depth
                use_depth_batch_mask = mask_add_depth.sum(dim=1) > 0
                if (~use_depth_batch_mask).sum() > 0 and N > 1:
                    pose_scale = poses_[..., 1:, :3, 3].norm(dim=-1)

                    static_threshold = 2e-2
                    is_static_mask = pose_scale.max(dim=1)[0] < static_threshold

                    pose_scale = pose_scale.mean(dim=1)
                    scale_aug = 0.8 + torch.rand((B,), device=device) * 0.4
                    pose_scale *= scale_aug

                    final_moving_mask = torch.logical_and(~use_depth_batch_mask, ~is_static_mask)
                    poses_[final_moving_mask, ..., :3, 3] /= (pose_scale.view(B, 1, 1)[final_moving_mask] + 1e-8)
                    normalized_depths[final_moving_mask] /= (pose_scale.view(B, 1, 1, 1)[final_moving_mask] + 1e-8)

                    dep_median[final_moving_mask] *= pose_scale[final_moving_mask]

                normalized_depths = normalized_depths.reshape(B*N, 1, H, W)

            if mask_add_depth.sum() > 0:
                depth_emb = self.depth_encoder(torch.cat([normalized_depths, depths_masks], dim=1), is_training=True)["x_norm_patchtokens"] + self.depth_emb
            else:
                depth_emb = torch.zeros_like(hidden)

            if mask_add_ray.sum() > 0:
                ray_emb = self.ray_embed(rays.reshape(B*N, H, W, 2).permute(0, 3, 1, 2))
            else:
                ray_emb = torch.zeros_like(hidden)

            use_depth_mask = mask_add_depth
            use_pose_mask = mask_add_pose

            hidden = hidden + ray_emb * mask_add_ray.reshape(B*N, 1, 1)
            hidden = hidden + depth_emb * mask_add_depth.reshape(B*N, 1, 1)

            return rgb_patch_tokens, hidden, poses_, use_depth_mask, use_pose_mask, dep_median
        
        return rgb_patch_tokens, hidden, None, None, None, None
    
    def _chunked_conv_head(self, head, feat, patch_h, patch_w, chunk_size=64):
        BN = feat.shape[0]
        if BN <= chunk_size:
            return head(feat, patch_h=patch_h, patch_w=patch_w)
        outputs = [[] for _ in range(len(head.output_block))] if isinstance(head.output_block, nn.ModuleList) else []
        for i in range(0, BN, chunk_size):
            chunk_out = head(feat[i:i+chunk_size], patch_h=patch_h, patch_w=patch_w)
            if isinstance(chunk_out, list):
                for j, o in enumerate(chunk_out):
                    outputs[j].append(o)
            else:
                outputs.append(chunk_out)
        if isinstance(outputs[0], list):
            return [torch.cat(parts, dim=0) for parts in outputs]
        return torch.cat(outputs, dim=0)

    def forward_head(self, hidden, pos, B, N, H, W, patch_h, patch_w, num_hand_tokens=0):
        device = hidden.device
        num_patch_tokens = patch_h * patch_w
        total_tokens = self.patch_start_idx + num_patch_tokens + num_hand_tokens
        patch_slice = slice(self.patch_start_idx, self.patch_start_idx + num_patch_tokens)

        # decode point
        ret_point = self.point_decoder(hidden, xpos=pos)

        # decode camera
        ret_camera = self.camera_decoder(hidden, xpos=pos)

        # decode metric
        pos_hw = pos.reshape(B, N * total_tokens, -1)
        ret_metric = self.metric_decoder(self.metric_token.repeat(B, 1, 1), hidden.reshape(B, N * total_tokens, -1), xpos=pos_hw[:, 0:1], ypos=pos_hw)

        # decode conf
        ret_conf = self.conf_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            point_feat = ret_point[:, patch_slice].float()
            xy, z = self._chunked_conv_head(self.point_head, point_feat, patch_h, patch_w)
            del point_feat
            xy = xy.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            z = z.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)

            z = torch.exp(z.clamp(max=15.0))
            local_points = torch.cat([xy * z, z], dim=-1)
            rays = F.normalize(torch.cat([xy, torch.ones_like(z)], dim=-1), dim=-1)

            camera_poses = self.camera_head(ret_camera[:, patch_slice].float(), patch_h, patch_w).reshape(B, N, 4, 4)

            metric = self.metric_head(ret_metric.float()).reshape(B).exp()

            # conf
            conf_feat = ret_conf[:, patch_slice].float()
            conf = self._chunked_conv_head(self.conf_head, conf_feat, patch_h, patch_w)[0]
            del conf_feat
            conf = conf.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)

            # points
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3] * metric.view(B, 1, 1, 1, 1)

            # convert camera poses to metric
            camera_poses[..., :3, 3] = camera_poses[..., :3, 3] * metric.view(B, 1, 1)

            # convert local_points to metric
            local_points = local_points * metric.view(B, 1, 1, 1, 1)

        return dict(
            points=points,
            local_points=local_points,
            rays=rays,
            conf=conf,
            camera_poses=camera_poses,  
            metric=metric,
        )


    def decode(self, hidden, N, H, W, poses, use_pose_mask, hand_tokens=None, hand_pos=None, object_tokens=None, object_pos=None):
        device = hidden.device

        if len(hidden.shape) == 4:
            B, N, hw, _ = hidden.shape
        else:
            BN, hw, _ = hidden.shape
            B = BN // N

        if hand_tokens is not None:
            if hand_pos is None:
                raise ValueError("hand_pos is required when hand_tokens is provided")
            if hand_tokens.shape[:2] != hidden.shape[:2]:
                raise ValueError("hand_tokens must align with hidden batch/view dimensions")
            hidden = torch.cat([hidden, hand_tokens], dim=2)
        if object_tokens is not None:
            if object_pos is None:
                raise ValueError("object_pos is required when object_tokens is provided")
            if object_tokens.shape[:2] != hidden.shape[:2]:
                raise ValueError("object_tokens must align with hidden batch/view dimensions")
            hidden = torch.cat([hidden, object_tokens], dim=2)
        hidden = hidden.reshape(B*N, -1, hidden.shape[-1])

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]
        pose_inject_blk_idx = 0

        pos = self.position_getter(B*N, H//self.patch_size, W//self.patch_size, hidden.device)
        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos_patch = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2, device=hidden.device, dtype=pos.dtype)
            if hand_tokens is not None:
                num_hand_tokens = hand_tokens.shape[2]
                pos_hand = hand_pos.reshape(B * N, num_hand_tokens, 2).to(hidden.device).to(pos.dtype) + 1
                pos = torch.cat([pos_special, pos_patch, pos_hand], dim=1)
                if object_tokens is not None:
                    num_object_tokens = object_tokens.shape[2]
                    pos_object = object_pos.reshape(B * N, num_object_tokens, 2).to(hidden.device).to(pos.dtype) + 1
                    pos = torch.cat([pos, pos_object], dim=1)
            else:
                pos = torch.cat([pos_special, pos_patch], dim=1)
                if object_tokens is not None:
                    num_object_tokens = object_tokens.shape[2]
                    pos_object = object_pos.reshape(B * N, num_object_tokens, 2).to(hidden.device).to(pos.dtype) + 1
                    pos = torch.cat([pos, pos_object], dim=1)

        if self.use_multimodal:
            if use_pose_mask.sum() == B * N:
                pose_inject_mask = None
            else:
                view_interaction_mask = use_pose_mask.unsqueeze(2) & use_pose_mask.unsqueeze(1)
                token_interaction_mask = view_interaction_mask.repeat_interleave(hw - self.patch_start_idx, dim=1)
                token_interaction_mask = token_interaction_mask.repeat_interleave(hw - self.patch_start_idx, dim=2)
                pose_inject_mask = token_interaction_mask[:, None]

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)

            if self.use_multimodal:
                if i in [1, 9, 17, 25, 33] and use_pose_mask.sum() > 0:
                    hidden = hidden.reshape(B, N, -1, 1024)
                    poses_feat = self.pose_inject_blk[pose_inject_blk_idx](hidden[..., self.patch_start_idx:, :].reshape(B, N*(hw-self.patch_start_idx), -1), poses, H, W, H//self.patch_size, W//self.patch_size, attn_mask=pose_inject_mask).reshape(B, N, -1, 1024)
                    hidden[..., self.patch_start_idx:, :] += poses_feat * use_pose_mask.view(B, N, 1, 1)
                    
                    hidden = hidden.reshape(B, N*hw, -1)
                    pose_inject_blk_idx += 1

            if i == len(self.decoder) - 2:
                temp_features = hidden.clone().reshape(B*N, hw, -1)

        concatenated = torch.cat((temp_features, hidden.reshape(B*N, hw, -1)), dim=-1)

        return concatenated, pos.reshape(B*N, hw, -1)
    
    
    def normalize_depth(self, depths: torch.Tensor, method: str = 'median') -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalizes a batch of depth maps using either median or mean normalization.

        Args:
            depths (torch.Tensor): A batch of depth maps with shape [B, N, H, W].
                                Non-positive values are treated as invalid depth data.
            method (str, optional): The normalization method to use.
                                    Can be 'median' or 'mean'. Defaults to 'median'.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The normalized depth maps.
                - The normalization factors (medians or means) used for each batch element.

        Raises:
            ValueError: If the method is not 'median' or 'mean'.
        """
        # 确保输入是 torch.Tensor
        if not isinstance(depths, torch.Tensor):
            depths = torch.tensor(depths, dtype=torch.float32)

        if method not in ['median', 'mean']:
            raise ValueError(f"Invalid normalization method: '{method}'. Choose 'median' or 'mean'.")

        B, N, H, W = depths.shape
        epsilon = 1e-8

        # Create a mask for valid depth values (positive values)
        valid_depths = torch.where(depths > 0, depths, float('nan'))
        valid_depths_reshaped = valid_depths.view(B, -1)

        if method == 'median':
            # Calculate the median for each depth map in the batch
            factors, _ = torch.nanmedian(valid_depths_reshaped, dim=1)
        elif method == 'mean':
            # Calculate the mean for each depth map in the batch
            factors = torch.nanmean(valid_depths_reshaped, dim=1)
        
        # Handle cases where all values might be NaN (e.g., all depths are 0 or negative)
        # In such cases, use 1.0 as the normalization factor to prevent division by zero.
        factors = torch.nan_to_num(factors, nan=1.0)
        
        # Reshape factors for broadcasting during division
        factors_for_division = factors.view(B, 1, 1, 1)

        # Perform normalization, adding a small epsilon to prevent division by zero
        normalized_depths = depths / (factors_for_division + epsilon)

        return normalized_depths, factors.reshape(-1)
