#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

mkdir -p "$PI3_MODEL_ROOT" "$HF_HOME"

ensure_python_package() {
    local package="$1"
    python - "$package" <<'PY' || python -m pip install -U "$package"
import importlib.util
import sys
name = sys.argv[1].replace("-", "_")
raise SystemExit(0 if importlib.util.find_spec(name) is not None else 1)
PY
}

ensure_python_package "huggingface_hub"
ensure_python_package "gdown"

download_hf_repo() {
    local repo_id="$1"
    local local_dir="$2"
    mkdir -p "$local_dir"
    huggingface-cli download "$repo_id" \
        --local-dir "$local_dir" \
        --local-dir-use-symlinks False
}

echo "[models] Downloading Pi3X to $PI3X_CKPT"
download_hf_repo "yyfz233/Pi3X" "$PI3X_CKPT"

HAMER_PARENT="$PI3_MODEL_ROOT/hamer"
HAMER_ARCHIVE="$HAMER_PARENT/hamer_demo_data.tar.gz"
mkdir -p "$HAMER_PARENT"

if [ ! -f "$HAMER_ENCODER_CKPT" ]; then
    echo "[models] Downloading HaMeR assets to $HAMER_PARENT"
    if command -v gdown >/dev/null 2>&1; then
        gdown "https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT" \
            -O "$HAMER_ARCHIVE" \
            --continue
    else
        python -m gdown "https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT" \
            -O "$HAMER_ARCHIVE" \
            --continue
    fi
    tar --warning=no-unknown-keyword --exclude=".*" -xzf "$HAMER_ARCHIVE" -C "$HAMER_PARENT"
else
    echo "[models] HaMeR checkpoint already exists: $HAMER_ENCODER_CKPT"
fi

echo "[models] Done"
echo "HAMER_CONFIG_FILE=$HAMER_CONFIG_FILE"
echo "HAMER_CACHE_DIR=$HAMER_CACHE_DIR"
echo "HAMER_ENCODER_CKPT=$HAMER_ENCODER_CKPT"
echo "PI3X_CKPT=$PI3X_CKPT"
