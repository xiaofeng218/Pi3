#!/usr/bin/env bash
set -euo pipefail

if [ -n "${ZSH_VERSION:-}" ]; then
    ASSET_REGISTRY_SOURCE="${(%):-%x}"
else
    ASSET_REGISTRY_SOURCE="${BASH_SOURCE[0]}"
fi

ASSET_REGISTRY_DIR="$(cd -- "$(dirname -- "$ASSET_REGISTRY_SOURCE")" && pwd)"
REPO_ROOT="$(cd -- "${ASSET_REGISTRY_DIR}/.." && pwd)"

export PI3_REPO_ROOT="${PI3_REPO_ROOT:-$REPO_ROOT}"
export PI3_ASSET_ROOT="${PI3_ASSET_ROOT:-$PI3_REPO_ROOT/data}"
export PI3_MODEL_ROOT="${PI3_MODEL_ROOT:-$PI3_ASSET_ROOT/model}"
export PI3_DATASET_ROOT="${PI3_DATASET_ROOT:-$PI3_ASSET_ROOT/dataset}"

export DEXYCB_ROOT="${DEXYCB_ROOT:-$PI3_DATASET_ROOT/dexycb}"
export DEX_YCB_DIR="${DEX_YCB_DIR:-$DEXYCB_ROOT}"

export HAMER_CONFIG_FILE="${HAMER_CONFIG_FILE:-$PI3_REPO_ROOT/configs/hamer/model_config.yaml}"
export HAMER_CACHE_DIR="${HAMER_CACHE_DIR:-$PI3_MODEL_ROOT/hamer/_DATA}"
export HAMER_ENCODER_CKPT="${HAMER_ENCODER_CKPT:-$HAMER_CACHE_DIR/hamer_ckpts/checkpoints/hamer.ckpt}"

export PI3X_CKPT="${PI3X_CKPT:-$PI3_MODEL_ROOT/pi3x/Pi3X}"
export HF_HOME="${HF_HOME:-$PI3_MODEL_ROOT/huggingface}"

export PYTHONPATH="$PI3_REPO_ROOT:$PI3_REPO_ROOT/dex-ycb-toolkit${PYTHONPATH:+:$PYTHONPATH}"
