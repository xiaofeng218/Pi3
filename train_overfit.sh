#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <experiment_name>}"
source asset_registry/env.sh
python scripts/train_pi3x.py --config-name overfit \
    "name=${EXPERIMENT}" \
    "log.output_dir=outputs/${EXPERIMENT}" \
    "log.ckpt_dir=outputs/${EXPERIMENT}/ckpts" \
