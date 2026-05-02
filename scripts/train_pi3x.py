from __future__ import annotations

import sys

sys.path.append(".")

import hydra

import trainers


@hydra.main(version_base="1.2", config_path="../configs", config_name="pi3x_hand_object")
def main(hydra_cfg):
    trainer = eval(hydra_cfg.trainer)(hydra_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
