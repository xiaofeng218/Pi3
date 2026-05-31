from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pi3.models.hamer.hand_mano_head import HandMANOHead
from pi3.models.layers.hand_global_head import HandGlobalHead
from pi3.models.layers.hand_pose_head import HandPoseHead
from pi3.models.layers.object_pose_head import ObjectPoseHead
class LocalTokenPredictionHeadTests(unittest.TestCase):
    def _make_cfg(self):
        class _Node(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        cfg = _Node()
        cfg.MODEL = _Node()
        cfg.MODEL.IMAGE_SIZE = 224
        cfg.MODEL.MANO_HEAD = _Node()
        cfg.MODEL.MANO_HEAD.JOINT_REP = "6d"
        cfg.EXTRA = _Node()
        cfg.EXTRA.FOCAL_LENGTH = 5000
        cfg.MANO = _Node()
        cfg.MANO.NUM_HAND_JOINTS = 15
        mean_params = Path(tempfile.gettempdir()) / "pi3x_local_token_hand_mean_params.npz"
        if not mean_params.is_file():
            np.savez(
                mean_params,
                pose=np.zeros((96,), dtype=np.float32),
                shape=np.zeros((10,), dtype=np.float32),
                cam=np.zeros((3,), dtype=np.float32),
            )
        cfg.MANO.MEAN_PARAMS = str(mean_params)
        return cfg

    def test_hand_global_head_outputs_rot_trans_and_scale(self) -> None:
        head = HandGlobalHead(in_dim=16, hidden_dim=8, patch_h=1, patch_w=3)
        pooled = torch.randn(4, 3, 16)

        out = head(pooled)

        self.assertEqual(tuple(out["pred_hand_global_orient"].shape), (4, 1, 3, 3))
        self.assertEqual(tuple(out["pred_hand_transl_dir"].shape), (4, 3))
        self.assertEqual(tuple(out["pred_hand_transl_scale"].shape), (4, 1))
        self.assertEqual(tuple(out["pred_hand_transl"].shape), (4, 3))
        self.assertEqual(tuple(out["pred_hand_scale"].shape), (4, 1))
        self.assertTrue(torch.allclose(out["pred_hand_transl"], out["pred_hand_transl_dir"] * out["pred_hand_transl_scale"]))

    def test_object_and_hand_heads_accept_shared_pooled_features(self) -> None:
        object_head = ObjectPoseHead(in_dim=16, hidden_dim=8, patch_h=1, patch_w=3)
        hand_global_head = HandGlobalHead(in_dim=16, hidden_dim=8, patch_h=1, patch_w=3)
        hand_pose_head = HandPoseHead(self._make_cfg(), in_dim=16, hidden_dim=8, patch_h=1, patch_w=3)
        hand_mano_head = HandMANOHead(self._make_cfg())

        hand_tokens = torch.randn(2, 3, 3, 16)
        object_tokens = torch.randn(2, 3, 3, 16)
        hand_valid = torch.tensor([[True, False, True], [True, True, False]], dtype=torch.bool)
        object_valid = torch.tensor([[False, True, True], [True, False, True]], dtype=torch.bool)
        hand_is_right = torch.tensor([[True, False, True], [True, True, False]], dtype=torch.bool)
        hand_betas = torch.zeros((2, 3, 10))

        object_out = object_head(object_tokens, valid_mask=object_valid)
        hand_global_out = hand_global_head(hand_tokens, valid_mask=hand_valid)
        hand_pose_out = hand_pose_head(hand_tokens, valid_mask=hand_valid)
        hand_mano_out = hand_mano_head(
            hand_pose=hand_pose_out["pred_hand_pose"].reshape(-1, 15, 3, 3),
            hand_is_right=hand_is_right.reshape(-1),
            hand_betas=hand_betas.reshape(-1, 10),
            global_orient=hand_global_out["pred_hand_global_orient"].reshape(-1, 1, 3, 3),
            transl=hand_global_out["pred_hand_transl"].reshape(-1, 3),
            scale=hand_global_out["pred_hand_scale"].reshape(-1, 1),
        )

        self.assertEqual(tuple(object_out["rot6d"].shape), (2, 3, 6))
        self.assertEqual(tuple(hand_mano_out["pred_mano_params"]["global_orient"].shape), (6, 1, 3, 3))
        self.assertEqual(tuple(hand_pose_out["pred_hand_pose"].shape), (2, 3, 15, 3, 3))
        self.assertTrue(torch.allclose(hand_mano_out["pred_mano_params"]["global_orient"], hand_global_out["pred_hand_global_orient"].reshape(-1, 1, 3, 3)))


if __name__ == "__main__":
    unittest.main()
