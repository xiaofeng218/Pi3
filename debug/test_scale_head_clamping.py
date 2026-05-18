from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from pi3.models.hamer.hand_mano_head import HandMANOHead
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

    def test_hand_mano_head_clamps_scale_exp(self) -> None:
        cfg = SimpleNamespace(
            MANO=SimpleNamespace(NUM_HAND_JOINTS=15, DATA_DIR=None, MEAN_PARAMS="/tmp/unused"),
            MODEL=SimpleNamespace(MANO_HEAD=SimpleNamespace(JOINT_REP="6d")),
        )
        head = HandMANOHead.__new__(HandMANOHead)
        torch.nn.Module.__init__(head)
        head.cfg = cfg
        head.in_dim = 4
        head.hidden_dim = 4
        head.joint_rep_type = "6d"
        head.joint_rep_dim = 6
        head.npose = 6 * (cfg.MANO.NUM_HAND_JOINTS + 1)
        head.project = torch.nn.Identity()
        head.decpose = torch.nn.Linear(4, head.npose)
        head.decshape = torch.nn.Linear(4, 10)
        head.dectransl_dir = torch.nn.Linear(4, 3)
        head.dectransl_scale = torch.nn.Linear(4, 1)
        head.decscale = torch.nn.Linear(4, 1)
        head.register_buffer("init_hand_pose", torch.zeros(1, head.npose))
        head.register_buffer("init_betas", torch.zeros(1, 10))
        head.mano = None
        with torch.no_grad():
            head.decpose.weight.zero_()
            valid_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            head.decpose.bias.copy_(valid_rot6d.repeat(cfg.MANO.NUM_HAND_JOINTS + 1))
            head.decshape.weight.zero_()
            head.decshape.bias.zero_()
            head.dectransl_dir.weight.zero_()
            head.dectransl_dir.bias.copy_(torch.tensor([0.0, 0.0, 1.0]))
            head.dectransl_scale.weight.zero_()
            head.dectransl_scale.bias.fill_(100.0)
            head.decscale.weight.zero_()
            head.decscale.bias.fill_(100.0)

        outputs = head(torch.zeros(2, 4))

        self.assertTrue(torch.isfinite(outputs["pred_hand_transl_scale"]).all())
        self.assertTrue(torch.isfinite(outputs["pred_hand_scale"]).all())


if __name__ == "__main__":
    unittest.main()
