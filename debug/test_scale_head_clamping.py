from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from pi3.models.hamer.hand_mano_head import HandMANOHead
from pi3.models.layers.hand_pose_head import HandPoseHead
from pi3.models.layers.object_pose_head import ObjectPoseHead


class ScaleHeadClampingTests(unittest.TestCase):
    def test_object_pose_head_clamps_scale_exp(self) -> None:
        head = ObjectPoseHead(in_dim=4, hidden_dim=4)
        object.__setattr__(head, "trunk", torch.nn.Identity())
        head.trans_scale_head = torch.nn.Linear(4, 1)
        head.scale_head = torch.nn.Linear(4, 1)
        with torch.no_grad():
            head.trans_scale_head.weight.zero_()
            head.trans_scale_head.bias.fill_(100.0)
            head.scale_head.weight.zero_()
            head.scale_head.bias.fill_(100.0)

        outputs = head(torch.zeros(2, 4))

        self.assertTrue(torch.isfinite(outputs["trans_scale"]).all())
        self.assertTrue(torch.isfinite(outputs["scale"]).all())

    def test_hand_pose_head_clamps_pose_output(self) -> None:
        cfg = SimpleNamespace(
            MANO=SimpleNamespace(NUM_HAND_JOINTS=15, DATA_DIR=None, MEAN_PARAMS="/tmp/unused"),
            MODEL=SimpleNamespace(MANO_HEAD=SimpleNamespace(JOINT_REP="6d")),
        )
        head = HandPoseHead.__new__(HandPoseHead)
        torch.nn.Module.__init__(head)
        head.cfg = cfg
        head.trunk = torch.nn.Identity()
        head.joint_rep_type = "6d"
        head.joint_rep_dim = 6
        head.npose = 6 * (cfg.MANO.NUM_HAND_JOINTS + 1)
        head.decpose = torch.nn.Linear(4, head.npose)
        head.register_buffer("init_hand_pose", torch.zeros(1, head.npose))
        with torch.no_grad():
            head.decpose.weight.zero_()
            valid_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            head.decpose.bias.copy_(valid_rot6d.repeat(cfg.MANO.NUM_HAND_JOINTS + 1))

        outputs = head(torch.zeros(2, 4))

        self.assertTrue(torch.isfinite(outputs["pred_hand_pose"]).all())
        self.assertTrue(torch.isfinite(outputs["pred_hand_global_orient"]).all())

    def test_hand_mano_head_uses_explicit_scale_without_prediction(self) -> None:
        cfg = SimpleNamespace(
            MANO=SimpleNamespace(NUM_HAND_JOINTS=15, DATA_DIR=None, MEAN_PARAMS="/tmp/unused"),
            MODEL=SimpleNamespace(MANO_HEAD=SimpleNamespace(JOINT_REP="6d")),
        )
        head = HandMANOHead(cfg, mano_layer=None)
        batch = 2
        hand_pose = torch.eye(3).view(1, 1, 3, 3).repeat(batch, 15, 1, 1)
        global_orient = torch.eye(3).view(1, 1, 3, 3).repeat(batch, 1, 1, 1)
        hand_betas = torch.zeros((batch, 10))
        transl = torch.zeros((batch, 3))
        scale = torch.exp(torch.full((batch, 1), 100.0))

        outputs = head(
            hand_pose=hand_pose,
            global_orient=global_orient,
            hand_betas=hand_betas,
            hand_is_right=torch.tensor([True, False], dtype=torch.bool),
            transl=transl,
            scale=scale,
        )

        self.assertTrue(torch.isfinite(outputs["pred_mano_params"]["hand_pose"]).all())
        self.assertTrue(torch.isfinite(outputs["pred_hand_mano_betas"]).all())


if __name__ == "__main__":
    unittest.main()
