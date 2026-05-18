#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${1:-outputs/full1.1}"
SUBJECT="${2:-20200709-subject-01}"
CHECKPOINT_NAME="${3:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

source asset_registry/env.sh

CKPT_ROOT="${EXPERIMENT_ROOT%/}/ckpts"
if [ ! -d "$CKPT_ROOT" ]; then
    echo "ERROR: checkpoint directory not found: $CKPT_ROOT" >&2
    exit 1
fi

if [ -n "$CHECKPOINT_NAME" ]; then
    CHECKPOINT_DIR="$CKPT_ROOT/$CHECKPOINT_NAME"
else
    CHECKPOINT_DIR="$(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'checkpoint-epoch-*' | sort | tail -n 1)"
fi

if [ -z "${CHECKPOINT_DIR:-}" ] || [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: no checkpoint found under $CKPT_ROOT" >&2
    exit 1
fi

OUTPUT_DIR="${EXPERIMENT_ROOT%/}/per_object_eval"
RERUN_DIR="${OUTPUT_DIR}/rerun_per_object"

echo "Running per-object object rotation evaluation"
echo "  experiment_root: $EXPERIMENT_ROOT"
echo "  checkpoint:      $CHECKPOINT_DIR"
echo "  subject:         $SUBJECT"
echo "  output_dir:      $OUTPUT_DIR"
echo "  rerun_dir:       $RERUN_DIR"
echo

python debug/eval_per_object_object_rot_loss.py \
    --checkpoint "$CHECKPOINT_DIR" \
    --subject "$SUBJECT" \
    --output-dir "$OUTPUT_DIR" \
    --export-rerun \
    --rerun-dir "$RERUN_DIR" \
    --progress-every 1
