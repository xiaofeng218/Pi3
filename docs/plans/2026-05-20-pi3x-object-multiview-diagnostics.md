# Pi3X Object Multiview Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an offline debug script that runs pretrained Pi3X image inference on DexYCB `object_multiview` RGB inputs, aligns predicted point clouds with predicted and GT camera poses, and exports rerun diagnostics to judge object-only behavior.

**Architecture:** Add one standalone debug script plus a small unit-test file. Reuse DexYCB dataset loading and Pi3 image-model forward behavior, but avoid changing trainer or HO production paths. Validate the pipeline in stages: GT depth export, prediction export, then combined rerun diagnostics.

**Tech Stack:** Python, PyTorch, DexYCB dataset loader, Pi3/Pi3X model code, rerun, pytest/unittest.

---

### Task 1: Lock down the existing model/data hooks

**Files:**
- Inspect: `debug/export-dexycb-batch-rrd.py`
- Inspect: `datasets/dexycb_dataset.py`
- Inspect: `pi3/models/pi3.py`
- Inspect: `trainers/pi3x_batch_utils.py`

**Step 1: Write the failing test**

Create a new test file skeleton that imports the future helper functions from the new debug script.

**Step 2: Run test to verify it fails**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py`
Expected: FAIL with import or missing symbol errors.

**Step 3: Write minimal implementation**

Create the new script with helper stubs only:

- object multiview extraction helper
- GT depth point-cloud helper
- predicted alignment helper

**Step 4: Run test to verify collection works**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py`
Expected: tests collect, then fail on assertions instead of import errors.

**Step 5: Commit**

```bash
git add debug/test_diagnose_pi3x_object_multiview_rerun.py debug/diagnose_pi3x_object_multiview_rerun.py
git commit -m "test: scaffold object multiview diagnostic coverage"
```

### Task 2: Build object multiview input extraction

**Files:**
- Create: `debug/diagnose_pi3x_object_multiview_rerun.py`
- Test: `debug/test_diagnose_pi3x_object_multiview_rerun.py`

**Step 1: Write the failing test**

Add a test that passes a synthetic `views[0]["object_multiview"]` payload and asserts the helper returns:

- `imgs` with shape `B x N x 3 x H x W`
- `depths` with shape `B x N x H x W`
- `intrinsics` with shape `B x N x 3 x 3`
- `camera_poses` with shape `B x N x 4 x 4`

**Step 2: Run test to verify it fails**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py -k extract`
Expected: FAIL on shape/content mismatch.

**Step 3: Write minimal implementation**

Implement a helper that reads the shared object multiview payload and returns typed tensors on CPU.

**Step 4: Run test to verify it passes**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py -k extract`
Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_diagnose_pi3x_object_multiview_rerun.py debug/diagnose_pi3x_object_multiview_rerun.py
git commit -m "feat: extract DexYCB object multiview inputs for diagnostics"
```

### Task 3: Build GT depth-to-world reference geometry

**Files:**
- Modify: `debug/diagnose_pi3x_object_multiview_rerun.py`
- Test: `debug/test_diagnose_pi3x_object_multiview_rerun.py`

**Step 1: Write the failing test**

Add a synthetic one-view test with identity intrinsics/pose and a simple depth map, asserting GT world points match expected coordinates after filtering valid depth pixels.

**Step 2: Run test to verify it fails**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py -k gt_world`
Expected: FAIL with wrong coordinates or missing filtering.

**Step 3: Write minimal implementation**

Implement a helper using existing geometry utilities to backproject depth maps and collect valid world-space point clouds with optional RGB colors.

**Step 4: Run test to verify it passes**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py -k gt_world`
Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_diagnose_pi3x_object_multiview_rerun.py debug/diagnose_pi3x_object_multiview_rerun.py
git commit -m "feat: add GT world pointcloud helper for diagnostics"
```

### Task 4: Build predicted alignment helpers

**Files:**
- Modify: `debug/diagnose_pi3x_object_multiview_rerun.py`
- Test: `debug/test_diagnose_pi3x_object_multiview_rerun.py`

**Step 1: Write the failing test**

Add tests for:

- flattening `pred.points` into a fused point cloud
- applying GT poses to `pred.local_points`
- preserving valid shapes across multiple views

**Step 2: Run test to verify it fails**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py -k pred`
Expected: FAIL with shape/alignment mismatch.

**Step 3: Write minimal implementation**

Implement helpers that:

- flatten predicted world points
- transform local points by GT poses
- optionally subsample large point clouds for rerun export

**Step 4: Run test to verify it passes**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py -k pred`
Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_diagnose_pi3x_object_multiview_rerun.py debug/diagnose_pi3x_object_multiview_rerun.py
git commit -m "feat: add predicted pointcloud alignment helpers"
```

### Task 5: Implement checkpoint/model loading

**Files:**
- Modify: `debug/diagnose_pi3x_object_multiview_rerun.py`
- Test: `debug/test_diagnose_pi3x_object_multiview_rerun.py`

**Step 1: Write the failing test**

Add a unit test around a tiny fake checkpoint/state-dict loader that verifies:

- the diagnostic chooses the Pi3 image path
- missing required image-model weights raise a clear error

**Step 2: Run test to verify it fails**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py -k checkpoint`
Expected: FAIL due to missing loader behavior.

**Step 3: Write minimal implementation**

Implement a helper that loads the pretrained Pi3/Pi3X image model from a checkpoint path and returns an eval-ready module on the target device.

**Step 4: Run test to verify it passes**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py -k checkpoint`
Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_diagnose_pi3x_object_multiview_rerun.py debug/diagnose_pi3x_object_multiview_rerun.py
git commit -m "feat: load pretrained Pi3 image model for diagnostics"
```

### Task 6: Implement rerun export assembly

**Files:**
- Modify: `debug/diagnose_pi3x_object_multiview_rerun.py`
- Test: `debug/test_diagnose_pi3x_object_multiview_rerun.py`

**Step 1: Write the failing test**

Add a test with a mocked rerun module asserting the exporter logs:

- per-view RGB/depth frames
- GT fused points
- predicted fused points
- GT and predicted camera transforms

**Step 2: Run test to verify it fails**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py -k rerun`
Expected: FAIL due to missing or incomplete logging.

**Step 3: Write minimal implementation**

Implement the export function and JSON manifest/statistics writing.

**Step 4: Run test to verify it passes**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py -k rerun`
Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_diagnose_pi3x_object_multiview_rerun.py debug/diagnose_pi3x_object_multiview_rerun.py
git commit -m "feat: export object multiview Pi3 diagnostics to rerun"
```

### Task 7: Add CLI and dataset iteration

**Files:**
- Modify: `debug/diagnose_pi3x_object_multiview_rerun.py`
- Test: `debug/test_diagnose_pi3x_object_multiview_rerun.py`

**Step 1: Write the failing test**

Add tests for parser defaults and sample-selection behavior:

- `--subject`
- `--mode`
- `--batch-index`
- `--sample-index`
- `--checkpoint`
- `--output-dir`

**Step 2: Run test to verify it fails**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py -k parser`
Expected: FAIL because parser or dispatch is incomplete.

**Step 3: Write minimal implementation**

Implement the CLI, dataset construction, collate path, sample selection, and main command wiring.

**Step 4: Run test to verify it passes**

Run: `pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py -k parser`
Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_diagnose_pi3x_object_multiview_rerun.py debug/diagnose_pi3x_object_multiview_rerun.py
git commit -m "feat: wire object multiview diagnostic CLI"
```

### Task 8: Run end-to-end validation on one real sample

**Files:**
- Run: `debug/diagnose_pi3x_object_multiview_rerun.py`
- Inspect: output directory under `outputs/`

**Step 1: Write the failing test**

No new unit test. This is verification against real assets.

**Step 2: Run command to verify baseline behavior**

Run:

```bash
python debug/diagnose_pi3x_object_multiview_rerun.py \
  --checkpoint outputs/full1.1/ckpts/checkpoint-epoch-0028 \
  --subject 20200709-subject-01 \
  --mode test \
  --batch-index 0 \
  --sample-index 0 \
  --output-dir outputs/debug_object_multiview_diag
```

Expected:

- command completes without exception
- `.rrd`, `meta.json`, and `stats.json` are written
- statistics show finite values

**Step 3: Inspect output**

Confirm rerun layers exist and GT/pred fused clouds are both present.

**Step 4: Re-run targeted tests**

Run:

```bash
pytest -q debug/test_diagnose_pi3x_object_multiview_rerun.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_diagnose_pi3x_object_multiview_rerun.py debug/diagnose_pi3x_object_multiview_rerun.py docs/plans/2026-05-20-pi3x-object-multiview-diagnostics-design.md docs/plans/2026-05-20-pi3x-object-multiview-diagnostics.md
git commit -m "feat: add Pi3X object multiview diagnostic export"
```
