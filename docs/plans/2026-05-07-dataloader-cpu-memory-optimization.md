# DataLoader CPU Memory Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce CPU memory usage of the DexYCB training dataloader by removing avoidable eager materialization and limiting worker-side batch amplification.

**Architecture:** Keep the current dataset contract stable where possible, but separate lightweight metadata from heavyweight tensor payloads. The main change is to stop building canonical object multiview tensors eagerly inside `DexYCBDataset.__getitem__`, add explicit config switches for expensive fields, and verify memory deltas with reproducible instrumentation.

**Tech Stack:** Python, PyTorch `DataLoader`, NumPy, PIL, OpenCV, local debug scripts under `debug/`

---

## Baseline Findings

Current measured behavior under the active DexYCB config:

- Dataset construction is cheap: main-process RSS rose from about `583.1MB` to `584.1MB`.
- One fetched sample is about `23.1MB`.
- `view[0]` is about `12.3MB`, with `object_multiview` alone about `10.8MB`.
- `view[1..7]` are about `1.57MB` each.
- `num_workers=0`: first batch raised main RSS to about `670.9MB`.
- `num_workers=2`, `prefetch_factor=1`, `pin_memory=True`: total PSS rose from about `411MB` to about `905MB` after the first fetched batch.

Largest measured payloads inside one sample:

| Field | Approx size |
|---|---:|
| `object_multiview.img` | `4.59MB` |
| `object_multiview.pts3d` | `4.59MB` |
| `object_multiview.depthmap` | `1.53MB` |
| per-view `pts3d` | `0.57MB` |
| per-view `img` | `0.57MB` |
| per-view `depthmap` | `0.19MB` |

Relevant code:

- `datasets/dexycb_dataset.py:342-382`
- `datasets/dexycb_dataset.py:475-580`
- `datasets/base/base_dataset.py:219-315`
- `datasets/__init__.py:117-133`
- `configs/data/dexycb_hand_object.yaml:3-23`
- `configs/train/train_pi3x_hand_object.yaml:5-9`

## Target Reduction Strategy

Priority order:

1. Stop eager canonical object multiview tensor construction in the dataloader.
2. Make per-view `pts3d` generation configurable and default it off for hand-object training if the trainer does not require CPU-side `pts3d`.
3. Add a memory-lean training preset that disables worker amplification during debugging and constrained runs.
4. Add reproducible instrumentation so regressions are visible.

Expected reductions:

- Remove eager `object_multiview.img/depthmap/pts3d`: save about `10.7MB` per sample.
- Remove per-view `pts3d` if not required: save about `4.6MB` per 8-frame sample.
- Combined sample reduction: from about `23.1MB` to about `7.8MB` to `12.4MB`, depending on whether per-view `pts3d` stays enabled.
- Worker-amplified memory should drop proportionally; with 2 workers and prefetched batches, practical total PSS reduction should be on the order of `200MB` to `450MB`.

### Task 1: Freeze the Memory Baseline in Scripts

**Files:**
- Modify: `debug/memory_profiler.py`
- Modify: `debug/memory_monitor.py`
- Create: `debug/profile_dexycb_loader_memory.py`

**Step 1: Write the failing test**

Create a smoke test script expectation in `debug/profile_dexycb_loader_memory.py` that prints:

- dataset-construction RSS
- first-batch sample payload size
- worker RSS/PSS/USS after iterator start

**Step 2: Run script to verify baseline output exists**

Run: `python debug/profile_dexycb_loader_memory.py --mode train --num-workers 0`

Expected: prints sample payload breakdown and exits successfully.

**Step 3: Write minimal implementation**

Add a dedicated profiling script that:

- instantiates `DexYCBDataset`
- recursively measures `numpy` and `torch` payload sizes
- optionally starts a dataloader iterator
- prints per-field sizes for `object_multiview` and regular views

**Step 4: Run script to verify it passes**

Run:

- `python debug/profile_dexycb_loader_memory.py --mode train --num-workers 0`
- `python debug/profile_dexycb_loader_memory.py --mode train --num-workers 2 --prefetch-factor 1`

Expected: both commands succeed and produce stable human-readable memory summaries.

**Step 5: Commit**

```bash
git add debug/memory_profiler.py debug/memory_monitor.py debug/profile_dexycb_loader_memory.py
git commit -m "chore: add DexYCB dataloader memory profiling"
```

### Task 2: Split Lightweight Object Metadata from Heavy Canonical Payload

**Files:**
- Modify: `datasets/dexycb_dataset.py:292-382`
- Modify: `datasets/dexycb_dataset.py:475-572`
- Test: `debug/test_dexycb_dataset_contract.py`

**Step 1: Write the failing test**

Add or extend dataset contract coverage to assert that when a new config flag disables heavy canonical payloads:

- `object_multiview` still contains lightweight metadata:
  - `template_vertices`
  - `normalization_center`
  - `normalization_scale`
  - `grasped_object_*` labels
- `object_multiview` does not contain:
  - `img`
  - `depthmap`
  - `pts3d`

**Step 2: Run test to verify it fails**

Run: `pytest debug/test_dexycb_dataset_contract.py -k object_multiview -v`

Expected: FAIL because no gating flag exists yet.

**Step 3: Write minimal implementation**

Add a dataset flag such as `include_object_multiview_payload` defaulting to `false` for training configs. Refactor:

- `_load_object_multiview_bundle()` to keep path and camera metadata only
- `_prepare_object_multiview()` to only build heavy tensors when the flag is enabled
- `_get_views()` to always emit the lightweight object metadata block

Do not change the lightweight keys already consumed by trainers unless a caller explicitly opts into the heavy payload.

**Step 4: Run test to verify it passes**

Run: `pytest debug/test_dexycb_dataset_contract.py -k object_multiview -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add datasets/dexycb_dataset.py debug/test_dexycb_dataset_contract.py
git commit -m "feat: gate heavy DexYCB object multiview payload"
```

### Task 3: Make Per-View `pts3d` Optional

**Files:**
- Modify: `datasets/base/base_dataset.py:219-315`
- Modify: `datasets/dexycb_dataset.py:475-580`
- Test: `debug/test_dexycb_dataset_contract.py`
- Test: `debug/test_pi3x_dexycb_object_integration.py`

**Step 1: Write the failing test**

Add a dataset-level flag such as `compute_pts3d_in_loader`. Tests should verify:

- when enabled, current behavior remains unchanged
- when disabled, returned views omit `pts3d` and `valid_mask`, or defer them in a way the trainer can reconstruct later

If the trainer requires these fields, tests must verify reconstruction happens exactly once in the trainer path instead.

**Step 2: Run test to verify it fails**

Run:

- `pytest debug/test_dexycb_dataset_contract.py -k pts3d -v`
- `pytest debug/test_pi3x_dexycb_object_integration.py -v`

Expected: FAIL until the flag and integration path exist.

**Step 3: Write minimal implementation**

Refactor `BaseDataset.__getitem__` so CPU-side `depthmap_to_absolute_camera_coordinates()` is conditional. If disabled:

- keep `depthmap`, `camera_intrinsics`, `camera_pose`, `z_far`
- defer `pts3d` materialization to the earliest consumer that truly needs it

Prefer a narrow change over broad batch-format churn.

**Step 4: Run test to verify it passes**

Run:

- `pytest debug/test_dexycb_dataset_contract.py -k pts3d -v`
- `pytest debug/test_pi3x_dexycb_object_integration.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add datasets/base/base_dataset.py datasets/dexycb_dataset.py debug/test_dexycb_dataset_contract.py debug/test_pi3x_dexycb_object_integration.py
git commit -m "feat: make loader-side pts3d generation optional"
```

### Task 4: Add a Memory-Lean Training Preset

**Files:**
- Modify: `configs/data/dexycb_hand_object.yaml`
- Modify: `configs/train/train_pi3x_hand_object.yaml`
- Create: `configs/train/train_pi3x_hand_object_memory_lean.yaml`
- Test: `debug/test_pi3x_config_smoke.py`

**Step 1: Write the failing test**

Add config smoke assertions for a memory-lean preset:

- `train.num_workers=0`
- `train_dataloader.prefetch_factor=1`
- `train_dataloader.persistent_workers=false`
- heavy object multiview payload disabled
- optional loader-side `pts3d` disabled

**Step 2: Run test to verify it fails**

Run: `pytest debug/test_pi3x_config_smoke.py -v`

Expected: FAIL because the preset does not exist yet.

**Step 3: Write minimal implementation**

Keep the default training config conservative, and add an explicit memory-lean preset for debugging and constrained hosts.

**Step 4: Run test to verify it passes**

Run: `pytest debug/test_pi3x_config_smoke.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add configs/data/dexycb_hand_object.yaml configs/train/train_pi3x_hand_object.yaml configs/train/train_pi3x_hand_object_memory_lean.yaml debug/test_pi3x_config_smoke.py
git commit -m "feat: add memory-lean Pi3X hand-object training preset"
```

### Task 5: Verify End-to-End Memory Reduction

**Files:**
- Modify: `debug/profile_dexycb_loader_memory.py`
- Optional docs update: `README.md` or `NEW_SERVER_SETUP.md`

**Step 1: Write the failing test**

Define verification thresholds in the profiling script output:

- sample payload reduced by at least `40%`
- total PSS under `num_workers=2` reduced materially versus baseline

**Step 2: Run baseline and optimized profiles**

Run:

- `python debug/profile_dexycb_loader_memory.py --mode train --num-workers 0`
- `python debug/profile_dexycb_loader_memory.py --mode train --num-workers 2 --prefetch-factor 1`

Expected: optimized config shows lower sample payload and lower worker-amplified PSS.

**Step 3: Document final measured deltas**

Record:

- baseline sample MB
- optimized sample MB
- baseline total PSS
- optimized total PSS

**Step 4: Run regression tests**

Run:

- `pytest debug/test_dexycb_dataset_contract.py -v`
- `pytest debug/test_pi3x_dexycb_object_integration.py -v`
- `pytest debug/test_pi3x_config_smoke.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add debug/profile_dexycb_loader_memory.py README.md NEW_SERVER_SETUP.md
git commit -m "docs: record dataloader memory optimization results"
```

## Recommended Implementation Order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5

## Risk Notes

- The biggest risk is breaking trainer code that implicitly expects `object_multiview.img/depthmap/pts3d` or per-view `pts3d` to always exist.
- The safest path is feature-gating, not deleting payloads outright.
- If any downstream module consumes canonical object tensors only for a subset of losses, move materialization to that loss path instead of the dataset.

## Expected Outcome

After Tasks 2 through 4:

- Sample payload should drop from about `23.1MB` to roughly `7.8MB` to `12.4MB`.
- Worker-amplified CPU memory should drop materially, especially in the `num_workers=2` path.
- Dataset initialization cost should remain effectively unchanged.
- Memory profiling should become a repeatable guardrail for future feature work.
