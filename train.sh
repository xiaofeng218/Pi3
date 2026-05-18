#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <experiment_name>}"
source asset_registry/env.sh

NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NPROC}" -eq 0 ]; then
    echo "ERROR: No GPUs found via nvidia-smi" >&2
    exit 1
fi
echo "Using ${NPROC} GPU(s)"

torchrun --nproc_per_node="${NPROC}" scripts/train_pi3x.py --config-name pi3x_hand_object \
    "name=${EXPERIMENT}" \
    "log.output_dir=outputs/${EXPERIMENT}" \
    "log.ckpt_dir=outputs/${EXPERIMENT}/ckpts" \
    # "train.iters_per_epoch=5" \
    # "test.iters_per_test=5"
