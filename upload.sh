#!/usr/bin/env bash
set -euo pipefail

FOREHOI_SRC=/mnt/2/data/dataset/ForeHOI/outputs/pi3x-pipeline
MODEL_SRC=/mnt/2/projects/Pi3/data/model
DART_SRC=/mnt/2/data/dataset/ForeHOI/DART

# Tar archives are written to /mnt/data (50G free) to avoid filling /mnt/2.
# Each archive is deleted immediately after a successful upload.
TAR_TMP=/mnt/data

echo "=========================================="
echo "  Pi3 R2 Upload"
echo "=========================================="
echo ""

# Helper: compress → upload → delete (one at a time to preserve disk space).
# Usage: tar_upload <source_parent> <subdir> <r2_dest>
tar_upload() {
    local parent="$1"
    local subdir="$2"
    local r2_dest="$3"
    local archive="$TAR_TMP/${subdir}.tar"

    echo "  Compressing ${subdir}/ → ${archive} ..."
    tar -cf "$archive" -C "$parent" "$subdir"
    echo "  Uploading ${archive} → ${r2_dest} ..."
    rclone copyto "$archive" "$r2_dest" \
      --progress \
      --s3-chunk-size 256M
    echo "  Deleting local archive ${archive} ..."
    rm "$archive"
}

echo "[1/8] Uploading HaMeR weights (~12G)..."
rclone copy "$MODEL_SRC/hamer/_DATA" r2:pi3-models/hamer \
  --progress --transfers 4 --checkers 8
echo "[1/8] Done: HaMeR weights"
echo ""

echo "[2/8] Uploading Pi3X weights (~5G)..."
rclone copy "$MODEL_SRC/pi3x" r2:pi3-models/pi3x \
  --progress --transfers 4
echo "[2/8] Done: Pi3X weights"
echo ""

echo "[3/8] ForeHOI render → render.tar (~35G, 93k files)..."
tar_upload "$FOREHOI_SRC" render r2:pi3-forehoi/render.tar
echo "[3/8] Done: ForeHOI render"
echo ""

echo "[4/8] Uploading pi3x_raw metadata (skipping symlinks)..."
rclone copy "$FOREHOI_SRC/pi3x_raw" r2:pi3-forehoi/pi3x_raw \
  --skip-links \
  --progress --transfers 4
echo "[4/8] Done: pi3x_raw"
echo ""

echo "[5/8] Uploading object_multiview_pyrender (~182M)..."
rclone copy "$FOREHOI_SRC/object_multiview_pyrender" r2:pi3-forehoi/object_multiview_pyrender \
  --progress --transfers 8
echo "[5/8] Done: object_multiview_pyrender"
echo ""

echo "[6/8] object_parts → object_parts.tar (~2G)..."
tar_upload "$FOREHOI_SRC" object_parts r2:pi3-forehoi/object_parts.tar
echo "[6/8] Done: object_parts"
echo ""

echo "[7/8] pipeline cache → cache.tar (~5.8G)..."
tar_upload "$FOREHOI_SRC" cache r2:pi3-forehoi/cache.tar
echo "[7/8] Done: pipeline cache"
echo ""

echo "[8/8] Uploading DART texture&accessories (~1.9G)..."
rclone copy "$DART_SRC/texture&accessories" r2:pi3-forehoi-deps/dart-texture-accessories \
  --progress --transfers 4
echo "[8/8] Done: DART texture&accessories"
echo ""

echo "=========================================="
echo "  Verifying upload"
echo "=========================================="
echo "--- Model weights ---"
rclone size r2:pi3-models/hamer
rclone size r2:pi3-models/pi3x
echo "--- ForeHOI tar archives ---"
rclone lsl r2:pi3-forehoi/ | grep -E "\.tar$"
echo "--- ForeHOI directories ---"
rclone size r2:pi3-forehoi/pi3x_raw
rclone size r2:pi3-forehoi/object_multiview_pyrender
echo "--- DART ---"
rclone size r2:pi3-forehoi-deps/dart-texture-accessories
echo ""
echo "All uploads complete."
