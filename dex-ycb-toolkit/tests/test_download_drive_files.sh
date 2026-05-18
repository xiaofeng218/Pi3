#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

FAKEBIN="$TMPDIR/fakebin"
OUTDIR="$TMPDIR/out"
INPUT="$TMPDIR/ids.txt"
STATE="$TMPDIR/state"
mkdir -p "$FAKEBIN" "$OUTDIR"
cat >"$INPUT" <<'EOF'
fake-file-id-1
fake-file-id-2
EOF

PAYLOAD1="$TMPDIR/payload1.bin"
PAYLOAD2="$TMPDIR/payload2.bin"
printf 'dexycb-test-payload-1\n' >"$PAYLOAD1"
printf 'dexycb-test-payload-2\n' >"$PAYLOAD2"
PAYLOAD1_SIZE="$(wc -c <"$PAYLOAD1" | tr -d ' ')"
PAYLOAD2_SIZE="$(wc -c <"$PAYLOAD2" | tr -d ' ')"
PAYLOAD1_MD5="$(md5sum "$PAYLOAD1" | awk '{print $1}')"
PAYLOAD2_MD5="$(md5sum "$PAYLOAD2" | awk '{print $1}')"

cat >"$FAKEBIN/curl" <<EOF
#!/usr/bin/env bash
set -euo pipefail
args="\$*"
all_args=("\$@")
if [[ "\$args" == *"fields=name,size,md5Checksum,mimeType"* ]]; then
  file_id="\${args##*/files/}"
  file_id="\${file_id%%\\?*}"
  if [[ "\$file_id" == "fake-file-id-1" ]]; then
    printf '{"name":"archive1.tar.gz","size":"%s","md5Checksum":"%s","mimeType":"application/gzip"}' "$PAYLOAD1_SIZE" "$PAYLOAD1_MD5"
  else
    printf '{"name":"archive2.tar.gz","size":"%s","md5Checksum":"%s","mimeType":"application/gzip"}' "$PAYLOAD2_SIZE" "$PAYLOAD2_MD5"
  fi
  exit 0
fi

out=""
while [[ \$# -gt 0 ]]; do
  case "\$1" in
    -o)
      out="\$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "\$out" ]]; then
  echo "missing -o" >&2
  exit 1
fi

file_id=""
for arg in "\${all_args[@]}"; do
  if [[ "\$arg" == *"alt=media"* ]]; then
    file_id="\${arg##*/files/}"
    file_id="\${file_id%%\\?*}"
  fi
done

attempt_file="$TMPDIR/\${file_id}.attempts"
attempt=0
if [[ -f "\$attempt_file" ]]; then
  attempt="\$(cat "\$attempt_file")"
fi
attempt="\$((attempt + 1))"
printf '%s' "\$attempt" >"\$attempt_file"

if [[ "\$attempt" -eq 1 ]]; then
  printf 'partial' >"\$out"
  echo "simulated transient curl failure" >&2
  exit 56
fi

sleep 2
if [[ "\$file_id" == "fake-file-id-1" ]]; then
  cat "$PAYLOAD1" >"\$out"
else
  cat "$PAYLOAD2" >"\$out"
fi
EOF
chmod +x "$FAKEBIN/curl"

cat >"$FAKEBIN/aria2c" <<'EOF'
#!/usr/bin/env bash
echo "aria2c must not be used by the downloader" >&2
exit 99
EOF
chmod +x "$FAKEBIN/aria2c"

start_time="$(date +%s)"
PATH="$FAKEBIN:$PATH" \
ACCESS_TOKEN="test-token" \
INPUT_FILE="$INPUT" \
OUTPUT_DIR="$OUTDIR" \
MAX_RETRIES=2 \
RETRY_DELAY_SECONDS=0 \
PARALLEL_DOWNLOADS=2 \
bash "$REPO_ROOT/download_drive_files.sh" >"$STATE.stdout" 2>"$STATE.stderr"
elapsed="$(( $(date +%s) - start_time ))"

test -f "$OUTDIR/archive1.tar.gz"
test -f "$OUTDIR/archive2.tar.gz"
test ! -f "$OUTDIR/archive1.tar.gz.part"
test ! -f "$OUTDIR/archive2.tar.gz.part"
cmp "$PAYLOAD1" "$OUTDIR/archive1.tar.gz"
cmp "$PAYLOAD2" "$OUTDIR/archive2.tar.gz"
test "$(cat "$TMPDIR/fake-file-id-1.attempts")" = "2"
test "$(cat "$TMPDIR/fake-file-id-2.attempts")" = "2"
test "$elapsed" -lt 4

echo "ok"
