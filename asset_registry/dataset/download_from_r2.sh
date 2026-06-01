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
        --skip-models)   SKIP_MODELS=1 ;;
        --skip-forehoi)  SKIP_FOREHOI=1 ;;
        --include-cache) INCLUDE_CACHE=1 ;;
    esac
done

if ! command -v rclone >/dev/null 2>&1; then
    echo "Error: rclone is not installed." >&2
    echo "Install: curl -L https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip && unzip /tmp/rclone.zip -d /tmp/rclone-tmp && cp /tmp/rclone-tmp/rclone-*-linux-amd64/rclone ~/bin/" >&2
    exit 1
fi

if ! rclone lsd "$R2:" >/dev/null 2>&1; then
    echo "Error: rclone remote '$R2' is not configured or not reachable." >&2
    echo "See docs/cloud-migration.md section 2 for setup instructions." >&2
    exit 1
fi

# Helper: stream a tar archive from R2 and extract to a target directory.
# Usage: extract_tar_from_r2 <r2_path> <extract_to_dir>
extract_tar_from_r2() {
    local r2_path="$1"
    local target_dir="$2"
    mkdir -p "$target_dir"
    echo "    Streaming $r2_path → $target_dir ..."
    rclone cat "$r2_path" | tar -xf - -C "$target_dir"
}

# ── Model weights ──────────────────────────────────────────────────────────────
if [[ "$SKIP_MODELS" -eq 0 ]]; then
    echo "==> Downloading HaMeR weights to $HAMER_CACHE_DIR ..."
    mkdir -p "$HAMER_CACHE_DIR"
    rclone copy "$R2:pi3-models/hamer" "$HAMER_CACHE_DIR" \
      --progress --transfers 4 --checkers 8
    echo "    Done: HaMeR weights"

    echo "==> Downloading Pi3X weights to $PI3X_CKPT ..."
    mkdir -p "$PI3X_CKPT"
    rclone copy "$R2:pi3-models/pi3x" "$PI3X_CKPT" \
      --progress --transfers 4
    echo "    Done: Pi3X weights"
fi

# ── ForeHOI generated data ─────────────────────────────────────────────────────
if [[ "$SKIP_FOREHOI" -eq 0 ]]; then
    echo "==> Extracting render.tar (~35G) to $FOREHOI_DATA_ROOT ..."
    extract_tar_from_r2 "$R2:pi3-forehoi/render.tar" "$FOREHOI_DATA_ROOT"
    echo "    Done: render/"

    echo "==> Downloading pi3x_raw metadata to $FOREHOI_DATA_ROOT/pi3x_raw ..."
    mkdir -p "$FOREHOI_DATA_ROOT/pi3x_raw"
    rclone copy "$R2:pi3-forehoi/pi3x_raw" "$FOREHOI_DATA_ROOT/pi3x_raw" \
      --progress --transfers 4
    echo "    Done: pi3x_raw/"

    echo "==> Downloading object_multiview_pyrender to $FOREHOI_DATA_ROOT/object_multiview_pyrender ..."
    mkdir -p "$FOREHOI_DATA_ROOT/object_multiview_pyrender"
    rclone copy "$R2:pi3-forehoi/object_multiview_pyrender" "$FOREHOI_DATA_ROOT/object_multiview_pyrender" \
      --progress --transfers 8
    echo "    Done: object_multiview_pyrender/"

    echo "==> Extracting object_parts.tar (~2G) to $FOREHOI_DATA_ROOT ..."
    extract_tar_from_r2 "$R2:pi3-forehoi/object_parts.tar" "$FOREHOI_DATA_ROOT"
    echo "    Done: object_parts/"

    if [[ "$INCLUDE_CACHE" -eq 1 ]]; then
        echo "==> Extracting cache.tar (~5.8G) to $FOREHOI_DATA_ROOT ..."
        extract_tar_from_r2 "$R2:pi3-forehoi/cache.tar" "$FOREHOI_DATA_ROOT"
        echo "    Done: cache/"
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
