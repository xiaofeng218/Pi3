#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_FILE="${INPUT_FILE:-$SCRIPT_DIR/dexycb_urls.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/2/projects/Pi3/data/dataset/dexycb}"
ACCESS_TOKEN="${ACCESS_TOKEN:-ya29.a0AQvPyIM9aNvnPK0HHV-K97G8oyW0sDJKNVAG7tcNUqSaK84QF22iJ1KJp6jPhz7zYMudFo6AIUSYjIFxQw6ep3ozRtwd7NCyy3VJRLWpkUHVznjVWxwdNHBNbXva1n_c3OCEO73dksHGAsjTFBUSZecO1o6b_7Ei4pKhYaCaC0mNPdb37_dgddN1YJK8MqlcEJXSMxgaCgYKAQ0SARISFQHGX2MicunZirymaYQIAVcrjblUSw0206}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-5}"
PARALLEL_DOWNLOADS="${PARALLEL_DOWNLOADS:-5}"

json_val() {
    python3 -c "import sys,json; print(json.load(sys.stdin).get('$1', ''))" 2>/dev/null
}

extract_file_id() {
    local url="$1"
    if [[ "$url" =~ /d/([^/?]+) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    else
        printf '%s\n' "$url"
    fi
}

require_binary() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Required command not found: $1" >&2
        exit 1
    }
}

fetch_metadata() {
    local file_id="$1"
    curl -sS --fail-with-body \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        "https://www.googleapis.com/drive/v3/files/$file_id?fields=name,size,md5Checksum,mimeType&supportsAllDrives=true"
}

download_file() {
    local file_id="$1"
    local part_path="$2"
    curl -fL -C - \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -o "$part_path" \
        "https://www.googleapis.com/drive/v3/files/$file_id?alt=media&supportsAllDrives=true"
}

download_with_retries() {
    local file_id="$1"
    local part_path="$2"
    local attempt=1

    while true; do
        if download_file "$file_id" "$part_path"; then
            return 0
        fi

        if (( attempt >= MAX_RETRIES )); then
            echo "Download failed after $attempt attempt(s)" >&2
            return 1
        fi

        echo "Download attempt $attempt failed; retrying in ${RETRY_DELAY_SECONDS}s..." >&2
        sleep "$RETRY_DELAY_SECONDS"
        attempt=$((attempt + 1))
    done
}

validate_not_error_body() {
    local path="$1"
    local mime_type="$2"
    if [[ "$mime_type" == application/json* || "$mime_type" == text/html* ]]; then
        echo "Download returned unexpected mime type: $mime_type" >&2
        sed -n '1,20p' "$path" >&2 || true
        return 1
    fi

    if head -c 1 "$path" | LC_ALL=C grep -q '{'; then
        echo "Download returned JSON instead of file content" >&2
        sed -n '1,20p' "$path" >&2 || true
        return 1
    fi
}

validate_size() {
    local expected_size="$1"
    local path="$2"
    local actual_size
    actual_size="$(wc -c <"$path" | tr -d ' ')"
    if [[ -n "$expected_size" && "$actual_size" != "$expected_size" ]]; then
        echo "Size mismatch for $path: expected $expected_size bytes, got $actual_size" >&2
        return 1
    fi
}

validate_md5() {
    local expected_md5="$1"
    local path="$2"
    local actual_md5
    if [[ -z "$expected_md5" ]]; then
        return 0
    fi

    actual_md5="$(md5sum "$path" | awk '{print $1}')"
    if [[ "$actual_md5" != "$expected_md5" ]]; then
        echo "MD5 mismatch for $path: expected $expected_md5, got $actual_md5" >&2
        return 1
    fi
}

download_one() {
    local file_id="$1"
    local metadata
    local name
    local size
    local md5
    local mime_type
    local final_path
    local part_path

    metadata="$(fetch_metadata "$file_id")"
    name="$(printf '%s' "$metadata" | json_val name)"
    size="$(printf '%s' "$metadata" | json_val size)"
    md5="$(printf '%s' "$metadata" | json_val md5Checksum)"
    mime_type="$(printf '%s' "$metadata" | json_val mimeType)"

    if [[ -z "$name" || "$name" == "null" ]]; then
        echo "Failed to retrieve metadata for file ID: $file_id" >&2
        echo "$metadata" >&2
        return 1
    fi

    final_path="$OUTPUT_DIR/$name"
    part_path="$final_path.part"

    echo "Downloading '$name' via Drive API..."
    download_with_retries "$file_id" "$part_path"
    validate_not_error_body "$part_path" "$mime_type"
    validate_size "$size" "$part_path"
    validate_md5 "$md5" "$part_path"
    mv -f "$part_path" "$final_path"
    echo "Downloaded: $final_path"
}

wait_for_slot() {
    while (( $(jobs -rp | wc -l | tr -d ' ') >= PARALLEL_DOWNLOADS )); do
        wait -n || return 1
    done
}

wait_for_all_jobs() {
    local status=0
    local pid

    for pid in $(jobs -rp); do
        if ! wait "$pid"; then
            status=1
        fi
    done

    return "$status"
}

main() {
    local url
    local file_id
    local status=0

    require_binary curl
    require_binary python3
    require_binary md5sum

    if [[ -z "$ACCESS_TOKEN" ]]; then
        echo "ACCESS_TOKEN is required" >&2
        exit 1
    fi

    if [[ ! -f "$INPUT_FILE" ]]; then
        echo "Input file '$INPUT_FILE' not found!" >&2
        exit 1
    fi

    if ! [[ "$PARALLEL_DOWNLOADS" =~ ^[1-9][0-9]*$ ]]; then
        echo "PARALLEL_DOWNLOADS must be a positive integer" >&2
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"

    while IFS= read -r url || [[ -n "$url" ]]; do
        [[ -z "$url" ]] && continue
        file_id="$(extract_file_id "$url")"
        wait_for_slot || status=1
        echo "Processing file ID: $file_id"
        download_one "$file_id" &
    done < "$INPUT_FILE"

    wait_for_all_jobs || status=1
    return "$status"
}

main "$@"
