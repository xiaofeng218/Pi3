#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 可视化数据
export DEX_YCB_DIR="${DEX_YCB_DIR:-/data/hanxiaofeng/dataset/dexycb}"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/manopth${PYTHONPATH:+:$PYTHONPATH}"

python -m examples.prepare_sequence_for_vis_upload \
    --camera-dir "$DEX_YCB_DIR/20200709-subject-01/20200709_141754/932122061900" \
    --output-root "$SCRIPT_DIR/output/prepared_for_upload"
