#!/usr/bin/env bash

set -euo pipefail

DATASET_ROOT="${1:-/data/hanxiaofeng/dataset/dexycb}"
OUTPUT_SUBDIR="${2:-canonical_views_224}"
FOV_Y_DEGREES="${3:-45}"
IMAGE_SIZE="${4:-224}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

python -m dex_ycb_toolkit.object_canonical_views \
    --dataset-root "${DATASET_ROOT}" \
    --image-size "${IMAGE_SIZE}" \
    --output-subdir "${OUTPUT_SUBDIR}" \
    --fov-y-degrees "${FOV_Y_DEGREES}" \
    --pyopengl-platform "${PYOPENGL_PLATFORM}" \
    --overwrite
