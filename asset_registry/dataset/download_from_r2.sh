#!/usr/bin/env bash
# Download ForeHOI pi3x-pipeline generated data and model weights from
# Cloudflare R2 to the current server.
#
# Prerequisites:
#   - rclone installed and configured with a remote named "r2"
#     (see docs/cloud-migration.md section 2 for setup instructions)
#   - asset_registry/env.sh sourced or PI3_DATA_ROOT set
#
# Usage:
#   export FOREHOI_DATA_ROOT=/data/forehoi   # where to put ForeHOI outputs
#   bash asset_registry/dataset/download_from_r2.sh
#
# Optional flags:
#   --skip-models       skip model weights download
#   --skip-forehoi      skip ForeHOI generated data download
#   --skip-cache        skip optional pipeline cache download
#   --include-cache     download pipeline cache (default: skipped)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

R2="${R2_REMOTE:-r2}"
FOREHOI_DATA_ROOT="${FOREHOI_DATA_ROOT:-$PI3_DATASET_ROOT/forehoi}"

SKIP_MODELS=0
SKIP_FOREHOI=0
INCLUDE_CACHE=0

for arg in "$@"; do
    case "$arg" in
        --skip-models)  SKIP_MODELS=1 ;;
        --skip-forehoi) SKIP_FOREHOI=1 ;;
        --include-cache) INCLUDE_CACHE=1 ;;
    esac
done

if ! command -v rclone >/dev/null 2>&1; then
    echo "Error: rclone is not installed. Run: curl https://rclone.org/install.sh | sudo bash" >&2
    exit 1
fi

if ! rclone lsd "$R2:" >/dev/null 2>&1; then
    echo "Error: rclone remote '$R2' is not configured or not reachable." >&2
    echo "See docs/cloud-migration.md section 2 for setup instructions." >&2
    exit 1
fi

# ── Model weights ──────────────────────────────────────────────────────────────
if [[ "$SKIP_MODELS" -eq 0 ]]; then
    echo "==> Downloading HaMeR weights to $HAMER_CACHE_DIR ..."
    mkdir -p "$HAMER_CACHE_DIR"
    rclone copy "$R2:pi3-models/hamer" "$HAMER_CACHE_DIR" \
      --progress --transfers 4 --checkers 8

    echo "==> Downloading Pi3X weights to $PI3X_CKPT ..."
    mkdir -p "$PI3X_CKPT"
    rclone copy "$R2:pi3-models/pi3x" "$PI3X_CKPT" \
      --progress --transfers 4
fi

# ── ForeHOI generated data ─────────────────────────────────────────────────────
if [[ "$SKIP_FOREHOI" -eq 0 ]]; then
    echo "==> Downloading ForeHOI render (~35G) to $FOREHOI_DATA_ROOT/render ..."
    mkdir -p "$FOREHOI_DATA_ROOT/render"
    rclone copy "$R2:pi3-forehoi/render" "$FOREHOI_DATA_ROOT/render" \
      --progress --transfers 8 --checkers 16 --multi-thread-streams 4

    echo "==> Downloading ForeHOI pi3x_raw (meta) to $FOREHOI_DATA_ROOT/pi3x_raw ..."
    mkdir -p "$FOREHOI_DATA_ROOT/pi3x_raw"
    rclone copy "$R2:pi3-forehoi/pi3x_raw" "$FOREHOI_DATA_ROOT/pi3x_raw" \
      --progress --transfers 4

    echo "==> Downloading object_multiview_pyrender to $FOREHOI_DATA_ROOT/object_multiview_pyrender ..."
    mkdir -p "$FOREHOI_DATA_ROOT/object_multiview_pyrender"
    rclone copy "$R2:pi3-forehoi/object_multiview_pyrender" "$FOREHOI_DATA_ROOT/object_multiview_pyrender" \
      --progress --transfers 8

    echo "==> Downloading object_parts to $FOREHOI_DATA_ROOT/object_parts ..."
    mkdir -p "$FOREHOI_DATA_ROOT/object_parts"
    rclone copy "$R2:pi3-forehoi/object_parts" "$FOREHOI_DATA_ROOT/object_parts" \
      --progress --transfers 4

    if [[ "$INCLUDE_CACHE" -eq 1 ]]; then
        echo "==> Downloading pipeline cache (~5.8G) to $FOREHOI_DATA_ROOT/cache ..."
        mkdir -p "$FOREHOI_DATA_ROOT/cache"
        rclone copy "$R2:pi3-forehoi/cache" "$FOREHOI_DATA_ROOT/cache" \
          --progress --transfers 4
    fi

    echo "==> Rebuilding pi3x_raw symlinks ..."
    bash "$SCRIPT_DIR/rebuild_forehoi_symlinks.sh" "$FOREHOI_DATA_ROOT"

    echo "==> Updating absolute paths in manifest.json ..."
    OLD_PREFIX="/mnt/2/data/dataset/ForeHOI/outputs/pi3x-pipeline"
    NEW_PREFIX="$FOREHOI_DATA_ROOT"
    sed -i "s|$OLD_PREFIX|$NEW_PREFIX|g" "$FOREHOI_DATA_ROOT/pi3x_raw/manifest.json"
    echo "    manifest.json paths updated: $OLD_PREFIX → $NEW_PREFIX"
fi

echo ""
echo "==> Download complete. Set the following environment variables:"
echo "    export FOREHOI_ROOT=$FOREHOI_DATA_ROOT/pi3x_raw"
echo "    export FOREHOI_OBJECT_MULTIVIEW_ROOT=$FOREHOI_DATA_ROOT/object_multiview_pyrender"
echo ""
echo "Then run the verification in docs/cloud-migration.md section 5.9."
