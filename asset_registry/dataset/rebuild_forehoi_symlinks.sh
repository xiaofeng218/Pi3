#!/usr/bin/env bash
# Rebuild symbolic links in pi3x_raw/ pointing to render/.
#
# pi3x_raw/<shard>/rgb|depth|mask/ contain per-frame symlinks that were
# skipped during rclone upload (--skip-links). This script recreates them
# on the target machine after downloading render/ and pi3x_raw/ from R2.
#
# Usage:
#   bash rebuild_forehoi_symlinks.sh <forehoi_data_root>
#
# where <forehoi_data_root> contains both pi3x_raw/ and render/.

set -euo pipefail

FOREHOI_DATA_ROOT="${1:-}"
if [[ -z "$FOREHOI_DATA_ROOT" ]]; then
    echo "Usage: $0 <forehoi_data_root>" >&2
    exit 1
fi

PI3X_RAW="$FOREHOI_DATA_ROOT/pi3x_raw"
RENDER="$FOREHOI_DATA_ROOT/render"

if [[ ! -d "$PI3X_RAW" ]]; then
    echo "Error: pi3x_raw directory not found: $PI3X_RAW" >&2
    exit 1
fi
if [[ ! -d "$RENDER" ]]; then
    echo "Error: render directory not found: $RENDER" >&2
    exit 1
fi

echo "[symlinks] Rebuilding symlinks in $PI3X_RAW ..."
echo "[symlinks] Render source: $RENDER"

count=0
skipped=0

for seq_dir in "$PI3X_RAW"/*/; do
    shard=$(basename "$seq_dir")

    # Skip manifest.json, split.json, etc.
    [[ -d "$seq_dir" ]] || continue

    render_seq="$RENDER/$shard"
    if [[ ! -d "$render_seq" ]]; then
        echo "  [skip] no render dir for $shard" >&2
        skipped=$((skipped + 1))
        continue
    fi

    for modality in rgb depth mask; do
        src_dir="$render_seq/$modality"
        dst_dir="$seq_dir/$modality"

        [[ -d "$src_dir" ]] || continue

        mkdir -p "$dst_dir"

        # Create one symlink per file found in render/<shard>/<modality>/
        while IFS= read -r -d '' frame_file; do
            fname=$(basename "$frame_file")
            dst_link="$dst_dir/$fname"
            if [[ ! -e "$dst_link" && ! -L "$dst_link" ]]; then
                ln -s "$frame_file" "$dst_link"
            fi
        done < <(find "$src_dir" -maxdepth 1 -type f -print0)
    done

    count=$((count + 1))
done

echo "[symlinks] Done. Linked $count sequences, skipped $skipped."
echo "[symlinks] Verify sample:"
sample_seq=$(ls "$PI3X_RAW" | grep "^large_" | head -1)
if [[ -n "$sample_seq" ]]; then
    ls -la "$PI3X_RAW/$sample_seq/rgb/" | head -4
fi
