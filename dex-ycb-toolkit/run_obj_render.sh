#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PI3_DATA_ROOT="${PI3_DATA_ROOT:-$REPO_ROOT/data}"
DATASET_ROOT="${1:-${DEXYCB_ROOT:-${DEX_YCB_DIR:-$PI3_DATA_ROOT/dataset/dexycb}}}"
OUTPUT_SUBDIR="${2:-canonical_views_224}"
FOV_Y_DEGREES="${3:-45}"
IMAGE_SIZE="${4:-224}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export DEX_YCB_DIR="${DEX_YCB_DIR:-$DATASET_ROOT}"
export MANO_ROOT="${MANO_ROOT:-$PI3_DATA_ROOT/model/hamer/_DATA/data/mano}"

python -m dex_ycb_toolkit.object_canonical_views \
    --dataset-root "${DATASET_ROOT}" \
    --image-size "${IMAGE_SIZE}" \
    --output-subdir "${OUTPUT_SUBDIR}" \
    --fov-y-degrees "${FOV_Y_DEGREES}" \
    --pyopengl-platform "${PYOPENGL_PLATFORM}" \
    --overwrite
