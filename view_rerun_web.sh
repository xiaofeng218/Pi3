#!/usr/bin/env bash
set -euo pipefail

RRD_PATH="${1:-}"
WEB_PORT="${RERUN_WEB_PORT:-9876}"
GRPC_PORT="${RERUN_GRPC_PORT:-9877}"

if [ -z "$RRD_PATH" ]; then
    echo "Usage: $0 <path/to/file.rrd>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


if [ ! -f "$RRD_PATH" ]; then
    echo "ERROR: .rrd file not found: $RRD_PATH" >&2
    exit 1
fi

if ! command -v rerun >/dev/null 2>&1; then
    echo "ERROR: rerun command not found in PATH" >&2
    exit 1
fi

echo "Starting rerun web viewer"
echo "  rrd: $RRD_PATH"
echo "  web viewer port: $WEB_PORT"
echo "  gRPC port: $GRPC_PORT"
echo
echo "Open on your local machine after SSH port forwarding:"
echo "  http://localhost:$WEB_PORT"
echo
echo "SSH config example:"
echo "Host pi3-rerun"
echo "    HostName <your_server>"
echo "    User <your_user>"
echo "    LocalForward $WEB_PORT localhost:$WEB_PORT"
echo "    LocalForward $GRPC_PORT localhost:$GRPC_PORT"
echo
echo "One-shot SSH command:"
echo "  ssh -L $WEB_PORT:localhost:$WEB_PORT -L $GRPC_PORT:localhost:$GRPC_PORT <your_user>@<your_server>"
echo

exec rerun "$RRD_PATH" \
    --web-viewer \
    --web-viewer-port "$WEB_PORT" \
    --port "$GRPC_PORT"
