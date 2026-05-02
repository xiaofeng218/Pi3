from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from pi3.models.pi3x import Pi3X


class Pi3XCheckpointLoadingTests(unittest.TestCase):
    def test_depth_encoder_bootstraps_from_encoder_without_checkpoint(self) -> None:
        model = Pi3X(use_multimodal=True, encoder_pretrained=False)
        encoder_state = model.encoder.state_dict()
        depth_state = model.depth_encoder.state_dict()

        self.assertTrue(
            torch.equal(
                depth_state["patch_embed.proj.weight"],
                encoder_state["patch_embed.proj.weight"][:, : depth_state["patch_embed.proj.weight"].shape[1]],
            ),
            msg="depth encoder patch embed should bootstrap from encoder even without ckpt",
        )

    def test_repo_id_checkpoint_loads_and_bootstraps_depth_encoder(self) -> None:
        model = Pi3X(use_multimodal=True)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as handle:
            checkpoint_path = Path(handle.name)
        try:
            torch.save({"state_dict": model.encoder.state_dict()}, checkpoint_path)

            with patch("pi3.models.pi3x.hf_hub_download", return_value=str(checkpoint_path)):
                loaded = Pi3X(use_multimodal=True, ckpt="yyfz233/Pi3X")
        finally:
            if checkpoint_path.exists():
                checkpoint_path.unlink()

        encoder_state = loaded.encoder.state_dict()
        depth_state = loaded.depth_encoder.state_dict()
        for key, tensor in depth_state.items():
            if key == "patch_embed.proj.weight":
                self.assertTrue(
                    torch.equal(tensor, encoder_state[key][:, : tensor.shape[1]]),
                    msg="depth patch embed should bootstrap from encoder channels",
                )
            else:
                self.assertTrue(torch.equal(tensor, encoder_state[key]), msg=f"mismatch at {key}")


if __name__ == "__main__":
    unittest.main()
