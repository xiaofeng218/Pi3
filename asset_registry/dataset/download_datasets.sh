#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

mkdir -p "$DEXYCB_ROOT"

echo "[dataset] Downloading DexYCB to $DEXYCB_ROOT"
bash "$PI3_REPO_ROOT/dex-ycb-toolkit/download_dexycb.sh" "$DEXYCB_ROOT"

echo "[dataset] Done"
echo "DEXYCB_ROOT=$DEXYCB_ROOT"
