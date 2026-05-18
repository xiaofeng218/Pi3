# Drive API Download Script Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current multi-connection Google Drive download flow with a single-connection `curl` flow that safely resumes downloads and verifies the downloaded file before promoting it to the final filename.

**Architecture:** Keep the script as a standalone Bash downloader driven by Drive file IDs. Fetch metadata from the Drive API first, download into `*.part` with `curl -L -C -`, validate size and MD5, reject JSON/HTML error bodies, and only then rename into place.

**Tech Stack:** Bash, `curl`, Python stdlib JSON parser, `md5sum`

---

### Task 1: Add a failing downloader behavior test

**Files:**
- Create: `tests/test_download_drive_files.sh`
- Test: `tests/test_download_drive_files.sh`

**Step 1: Write the failing test**

Add a Bash test that:
- stubs `curl`
- provides fake metadata with `name`, `size`, and `md5Checksum`
- simulates a binary download into `*.part`
- expects the script to promote the file only after size and MD5 validation

**Step 2: Run test to verify it fails**

Run: `bash tests/test_download_drive_files.sh`
Expected: FAIL because the current script is not structured for safe validation/promotion.

### Task 2: Rework the downloader script

**Files:**
- Modify: `download_drive_files.sh`
- Test: `tests/test_download_drive_files.sh`

**Step 1: Write minimal implementation**

Change the script to:
- use `set -euo pipefail`
- accept `ACCESS_TOKEN`, `INPUT_FILE`, and `OUTPUT_DIR` from env
- fetch Drive metadata before download
- download with single-connection `curl -L -C -` into `*.part`
- detect JSON/HTML error bodies
- validate file size and MD5
- rename `*.part` to final name only after validation passes

**Step 2: Run test to verify it passes**

Run: `bash tests/test_download_drive_files.sh`
Expected: PASS

### Task 3: Verify script surface behavior

**Files:**
- Modify: `download_drive_files.sh`
- Test: `tests/test_download_drive_files.sh`

**Step 1: Run focused verification**

Run:
- `bash tests/test_download_drive_files.sh`
- `bash -n download_drive_files.sh`

Expected:
- tests pass
- shell syntax check passes
