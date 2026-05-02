from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .geometry import aa_to_rotmat


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "r"):
        value = value.r
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.array(value, copy=True)


def _load_mano_model(mano_root: str | Path, side: str) -> dict[str, Any]:
    mano_root = Path(mano_root)
    mano_path = mano_root / ("MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl")
    if not mano_path.exists():
        raise FileNotFoundError(f"Could not find MANO model file: {mano_path}")

    with open(mano_path, "rb") as f:
        dd = pickle.load(f, encoding="latin1")

    return dd


def rotmat_to_axis_angle(rotmat: torch.Tensor) -> torch.Tensor:
    rotmat = rotmat.reshape(-1, 3, 3)
    trace = torch.diagonal(rotmat, dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)

    skew = torch.stack(
        [
            rotmat[:, 2, 1] - rotmat[:, 1, 2],
            rotmat[:, 0, 2] - rotmat[:, 2, 0],
            rotmat[:, 1, 0] - rotmat[:, 0, 1],
        ],
        dim=-1,
    )
    axis = skew / (2.0 * torch.sin(theta).unsqueeze(-1) + 1e-6)
    axis_angle = axis * theta.unsqueeze(-1)
    small = theta < 1e-4
    if small.any():
        axis_angle = axis_angle.clone()
        axis_angle[small] = 0.5 * skew[small]
    return axis_angle.reshape(*rotmat.shape[:-2], 3)


def _normalize_vector(v: torch.Tensor) -> torch.Tensor:
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(v_mag, v.new_tensor(1e-8))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    return v / v_mag


def _cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)


def _compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    x = _normalize_vector(x_raw)
    z = _cross_product(x, y_raw)
    z = _normalize_vector(z)
    y = _cross_product(z, x)
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    return torch.cat((x, y, z), 2)


def _robust_compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    x = _normalize_vector(x_raw)
    y = _normalize_vector(y_raw)
    middle = _normalize_vector(x + y)
    orthmid = _normalize_vector(x - y)
    x = _normalize_vector(middle + orthmid)
    y = _normalize_vector(middle - orthmid)
    z = _normalize_vector(_cross_product(x, y))
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    return torch.cat((x, y, z), 2)


def _posemap_axisang_impl(pose_vectors: torch.Tensor):
    rot_nb = int(pose_vectors.shape[1] / 3)
    pose_vec_reshaped = pose_vectors.contiguous().view(-1, 3)
    rot_mats = _batch_rodrigues(pose_vec_reshaped)
    rot_mats = rot_mats.view(pose_vectors.shape[0], rot_nb * 9)
    pose_maps = _subtract_flat_id_impl(rot_mats)
    return pose_maps, rot_mats


def _with_zeros_impl(tensor: torch.Tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new_tensor([0.0, 0.0, 0.0, 1.0])
    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    return torch.cat(concat_list, 1)


def _subtract_flat_id_impl(rot_mats: torch.Tensor):
    rot_nb = int(rot_mats.shape[1] / 9)
    id_flat = torch.eye(3, dtype=rot_mats.dtype, device=rot_mats.device).view(1, 9).repeat(rot_mats.shape[0], rot_nb)
    return rot_mats - id_flat


def _batch_rodrigues(axisang: torch.Tensor) -> torch.Tensor:
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    return _quat2mat(quat).view(-1, 9)


def _quat2mat(quat: torch.Tensor) -> torch.Tensor:
    norm_quat = quat / quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    batch_size = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(batch_size, 3, 3)
    return rotMat


class ManoLayer(nn.Module):
    __constants__ = [
        "use_pca",
        "rot",
        "ncomps",
        "kintree_parents",
        "check",
        "side",
        "center_idx",
        "joint_rot_mode",
    ]

    def __init__(
        self,
        center_idx=None,
        flat_hand_mean=True,
        ncomps=6,
        side="right",
        mano_root="mano/models",
        use_pca=True,
        root_rot_mode="axisang",
        joint_rot_mode="axisang",
        robust_rot=False,
    ):
        super().__init__()

        self.center_idx = center_idx
        self.robust_rot = robust_rot
        self.rot = 3 if root_rot_mode == "axisang" else 6
        self.flat_hand_mean = flat_hand_mean
        self.side = side
        self.is_rhand = side == "right"
        self.use_pca = use_pca
        self.joint_rot_mode = joint_rot_mode
        self.root_rot_mode = root_rot_mode
        self.ncomps = ncomps if use_pca else 45

        self.mano_root = Path(mano_root)
        dd = _load_mano_model(self.mano_root, side)

        hands_components = _to_numpy(dd["hands_components"])
        hands_mean = np.zeros(hands_components.shape[1], dtype=np.float32) if flat_hand_mean else _to_numpy(dd["hands_mean"])
        hands_mean = hands_mean.copy()

        # MANO pickles used by manopth do not store a `betas` entry; the mean
        # shape is zero-initialized and the learned/default shape comes from the
        # runtime `betas` input.
        self.register_buffer("th_betas", torch.zeros((1, 10), dtype=torch.float32))
        self.register_buffer("th_shapedirs", torch.as_tensor(_to_numpy(dd["shapedirs"]), dtype=torch.float32))
        self.register_buffer("th_posedirs", torch.as_tensor(_to_numpy(dd["posedirs"]), dtype=torch.float32))
        self.register_buffer("th_v_template", torch.as_tensor(_to_numpy(dd["v_template"]), dtype=torch.float32).unsqueeze(0))
        self.register_buffer("th_J_regressor", torch.as_tensor(_to_numpy(dd["J_regressor"]), dtype=torch.float32))
        self.register_buffer("th_weights", torch.as_tensor(_to_numpy(dd["weights"]), dtype=torch.float32))
        self.register_buffer("th_faces", torch.as_tensor(_to_numpy(dd["f"]), dtype=torch.long))
        self.faces = self.th_faces

        th_hands_mean = torch.as_tensor(hands_mean, dtype=torch.float32).unsqueeze(0)
        if self.use_pca or self.joint_rot_mode == "axisang":
            self.register_buffer("th_hands_mean", th_hands_mean)
            selected_components = hands_components[: self.ncomps]
            self.register_buffer("th_comps", torch.as_tensor(hands_components, dtype=torch.float32))
            self.register_buffer("th_selected_comps", torch.as_tensor(selected_components, dtype=torch.float32))
            self.register_buffer("_th_selected_comps_pinv", torch.linalg.pinv(torch.as_tensor(selected_components, dtype=torch.float32)))
        else:
            self.register_buffer("th_hands_mean_rotmat", aa_to_rotmat(th_hands_mean.view(15, 3)).reshape(15, 3, 3))

        kintree_table = _to_numpy(dd["kintree_table"])
        self.kintree_table = kintree_table
        self.kintree_parents = list(kintree_table[0].tolist())

    def _pose_coeffs_from_axis_angle(self, global_orient: torch.Tensor, hand_pose: torch.Tensor) -> torch.Tensor:
        batch_size = global_orient.shape[0]
        hand_pose = hand_pose.reshape(batch_size, -1)

        if self.use_pca:
            hand_delta = hand_pose - self.th_hands_mean.expand_as(hand_pose)
            hand_coeffs = hand_delta @ self._th_selected_comps_pinv
            return torch.cat([global_orient.reshape(batch_size, -1), hand_coeffs], dim=1)

        return torch.cat([global_orient.reshape(batch_size, -1), hand_pose], dim=1)

    def forward_axis_angle(
        self,
        global_orient: torch.Tensor,
        hand_pose: torch.Tensor,
        betas: torch.Tensor,
        th_trans: torch.Tensor | None = None,
        root_palm: torch.Tensor = torch.Tensor([0]),
        share_betas: torch.Tensor = torch.Tensor([0]),
    ):
        if global_orient.dim() != 2 or global_orient.shape[-1] != 3:
            raise ValueError("global_orient must have shape (B, 3)")
        if hand_pose.dim() != 3 or hand_pose.shape[-2:] != (15, 3):
            raise ValueError("hand_pose must have shape (B, 15, 3)")
        pose_coeffs = self._pose_coeffs_from_axis_angle(global_orient, hand_pose)
        return self.forward(
            pose_coeffs,
            th_betas=betas,
            th_trans=th_trans if th_trans is not None else torch.zeros((global_orient.shape[0], 3), device=global_orient.device, dtype=global_orient.dtype),
            root_palm=root_palm,
            share_betas=share_betas,
        )

    def forward_rotmat(
        self,
        global_orient: torch.Tensor,
        hand_pose: torch.Tensor,
        betas: torch.Tensor,
        th_trans: torch.Tensor | None = None,
        root_palm: torch.Tensor = torch.Tensor([0]),
        share_betas: torch.Tensor = torch.Tensor([0]),
    ):
        if global_orient.dim() not in (3, 4) or global_orient.shape[-2:] != (3, 3):
            raise ValueError("global_orient must have shape (B, 1, 3, 3) or (B, 3, 3)")
        if global_orient.dim() == 4:
            global_orient = global_orient[:, 0]
        if hand_pose.dim() != 4 or hand_pose.shape[-2:] != (3, 3):
            raise ValueError("hand_pose must have shape (B, 15, 3, 3)")
        global_aa = rotmat_to_axis_angle(global_orient)
        hand_aa = rotmat_to_axis_angle(hand_pose.reshape(-1, 3, 3)).reshape(hand_pose.shape[0], 15, 3)
        return self.forward_axis_angle(
            global_orient=global_aa,
            hand_pose=hand_aa,
            betas=betas,
            th_trans=th_trans,
            root_palm=root_palm,
            share_betas=share_betas,
        )

    def forward(
        self,
        th_pose_coeffs,
        th_betas=torch.zeros(1),
        th_trans=torch.zeros(1),
        root_palm=torch.Tensor([0]),
        share_betas=torch.Tensor([0]),
    ):
        batch_size = th_pose_coeffs.shape[0]
        if self.use_pca or self.joint_rot_mode == "axisang":
            th_hand_pose_coeffs = th_pose_coeffs[:, self.rot : self.rot + self.ncomps]
            if self.use_pca:
                th_full_hand_pose = th_hand_pose_coeffs.mm(self.th_selected_comps)
            else:
                th_full_hand_pose = th_hand_pose_coeffs

            th_full_pose = torch.cat([th_pose_coeffs[:, : self.rot], self.th_hands_mean + th_full_hand_pose], 1)
            if self.root_rot_mode == "axisang":
                th_pose_map, th_rot_map = _posemap_axisang_impl(th_full_pose)
                root_rot = th_rot_map[:, :9].view(batch_size, 3, 3)
                th_rot_map = th_rot_map[:, 9:]
                th_pose_map = th_pose_map[:, 9:]
            else:
                th_pose_map, th_rot_map = _posemap_axisang_impl(th_full_pose[:, 6:])
                root_rot = _compute_rot6d_root(th_full_pose[:, :6], robust_rot=self.robust_rot)
        else:
            assert th_pose_coeffs.dim() == 4
            assert th_pose_coeffs.shape[2:4] == (3, 3)
            th_pose_rots = _batch_rotprojs_impl(th_pose_coeffs)
            th_rot_map = th_pose_rots[:, 1:].view(batch_size, -1)
            th_pose_map = _subtract_flat_id_impl(th_rot_map)
            root_rot = th_pose_rots[:, 0]

        if th_betas is None or th_betas.numel() == 1:
            th_v_shaped = torch.matmul(self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(batch_size, 1, 1)
        else:
            if share_betas is not None and bool(torch.as_tensor(share_betas).item()):
                th_betas = th_betas.mean(0, keepdim=True).expand(th_betas.shape[0], 10)
            th_v_shaped = torch.matmul(self.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)

        th_v_posed = th_v_shaped + torch.matmul(self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_trans = _with_zeros_impl(torch.cat([root_rot, root_j], 2))

        all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = th_j[:, lev1_idxs]
        lev2_j = th_j[:, lev2_idxs]
        lev3_j = th_j[:, lev3_idxs]

        all_transforms = [root_trans.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = _with_zeros_impl(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4))

        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = _with_zeros_impl(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))

        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = _with_zeros_impl(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_results_global = th_results

        joint_js = torch.cat([th_j, th_j.new_zeros(th_j.shape[0], 16, 1)], 2)
        tmp2 = torch.matmul(th_results, joint_js.unsqueeze(3))
        th_results2 = (th_results - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1)

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))
        th_rest_shape_h = torch.cat(
            [
                th_v_posed.transpose(2, 1),
                torch.ones((batch_size, 1, th_v_posed.shape[1]), dtype=th_T.dtype, device=th_T.device),
            ],
            1,
        )

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = th_results_global[:, :, :3, 3]

        if self.side == "right":
            tips = th_verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = th_verts[:, [745, 317, 445, 556, 673]]
        if bool(root_palm):
            palm = (th_verts[:, 95] + th_verts[:, 22]).unsqueeze(1) / 2
            th_jtr = torch.cat([palm, th_jtr[:, 1:]], 1)
        th_jtr = torch.cat([th_jtr, tips], 1)
        th_jtr = th_jtr[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]

        if th_trans is None or bool(torch.norm(th_trans) == 0):
            if self.center_idx is not None:
                center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
                th_jtr = th_jtr - center_joint
                th_verts = th_verts - center_joint
        else:
            th_jtr = th_jtr + th_trans.unsqueeze(1)
            th_verts = th_verts + th_trans.unsqueeze(1)

        th_verts = th_verts * 1000
        th_jtr = th_jtr * 1000
        return type("MANOOutput", (), {"vertices": th_verts, "joints": th_jtr, "faces": self.th_faces})()


def _batch_rotprojs_impl(rotmat: torch.Tensor):
    proj_rotmats = []
    for batch_rotmats in rotmat:
        proj_batch_rotmats = []
        for rot in batch_rotmats:
            U, _, V = rot.cpu().svd()
            rot_proj = torch.matmul(U, V.transpose(0, 1))
            if rot_proj.det() < 0:
                rot_proj[:, 2] = -1 * rot_proj[:, 2]
            proj_batch_rotmats.append(rot_proj.to(device=rot.device))
        proj_rotmats.append(torch.stack(proj_batch_rotmats))
    return torch.stack(proj_rotmats)


def _compute_rot6d_root(root_coeffs: torch.Tensor, robust_rot: bool = False):
    if robust_rot:
        return _robust_compute_rotation_matrix_from_ortho6d(root_coeffs)
    return _compute_rotation_matrix_from_ortho6d(root_coeffs)


def build_mano_layer(side: str = "right", mano_root: str | Path = "mano/models", **kwargs) -> ManoLayer:
    return ManoLayer(side=side, mano_root=mano_root, **kwargs)


def build_mano_layer_pair(mano_root: str | Path = "mano/models", **kwargs) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "right": build_mano_layer(side="right", mano_root=mano_root, **kwargs),
            "left": build_mano_layer(side="left", mano_root=mano_root, **kwargs),
        }
    )
