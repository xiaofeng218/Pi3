from __future__ import annotations

import os
import sys

sys.path.append(".")
                
import hydra
import torch.multiprocessing as mp

import trainers


def _apply_mp_runtime_overrides() -> None:
    sharing_strategy = os.environ.get("PI3_MP_SHARING_STRATEGY", None)
    if sharing_strategy:
        mp.set_sharing_strategy(sharing_strategy)


@hydra.main(version_base="1.2", config_path="../configs", config_name="pi3x_hand_object")
def main(hydra_cfg):
    trainer = eval(hydra_cfg.trainer)(hydra_cfg)
    trainer.train()


if __name__ == "__main__":
    _apply_mp_runtime_overrides()
    main()
