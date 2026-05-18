# Module 6 Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove leftover legacy GT post-processing, debug/inspect dependencies, and obsolete logging/test expectations so the refactored training contract is the only supported path.

**Architecture:** Keep the training contract stable and do not change model semantics in this module. Remove code that only exists for deprecated fields (`hand_vertices`, `object_vertices_2d`, legacy `grasped_object_*`) or legacy trainer-side GT synthesis, then update debug/visualization/test consumers to match the new contract.

**Tech Stack:** Python, PyTorch, unittest, existing Pi3X trainer/visualization utilities.

---

### Task 1: Remove trainer-side hand GT post-processing

**Files:**
- Modify: `trainers/pi3x_trainer.py`
- Test: `debug/test_pi3x_trainer_smoke.py`

**Step 1: Write the failing test**

Add a smoke assertion that `forward_batch()` no longer relies on `hand_vertices`, `hand_global_orient_rotmat`, or `hand_pose_rotmat` in the returned `gt`.

**Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_pi3x_trainer_smoke`

Expected: FAIL because the trainer still writes deprecated hand GT fields.

**Step 3: Write minimal implementation**

Delete `_compute_hand_gt_mesh()` and `_decode_hand_gt_pose_rotmats()` if they are no longer referenced, and remove all call sites and `gt[...]` assignments for the deprecated hand fields.

**Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_pi3x_trainer_smoke`

Expected: PASS.

**Step 5: Commit**

```bash
git add trainers/pi3x_trainer.py debug/test_pi3x_trainer_smoke.py
git commit -m "refactor: drop trainer hand gt post-processing"
```

### Task 2: Remove trainer-side object 2D leftovers from inspection paths

**Files:**
- Modify: `debug/inspect_loss_components.py`
- Modify: `debug/pi3x_train_finite_check.py`
- Modify: `debug/check_pi3x_pred_mesh_selfcheck.py`
- Modify: `debug/inspect_pi3x_2d_projection_consistency.py`

**Step 1: Write the failing test**

Add or update assertions so these scripts do not require `object_2d_loss`, `object_vertices_2d`, or `object_vertices_2d_valid` to be present in training outputs.

**Step 2: Run test to verify it fails**

Run:
`python -m unittest debug.test_tensorboard_log_cleanup`
`python -m unittest debug.test_pi3x_pred_mesh_selfcheck`
`python -m unittest debug.test_inspect_pi3x_pred_vs_gt`

Expected: FAIL where old keys are still referenced.

**Step 3: Write minimal implementation**

Remove old key reads, guard optional display code behind the new contract, or replace it with derived values computed only when the corresponding debug script explicitly generates them.

**Step 4: Run test to verify it passes**

Run:
`python -m unittest debug.test_tensorboard_log_cleanup`
`python -m unittest debug.test_pi3x_pred_mesh_selfcheck`
`python -m unittest debug.test_inspect_pi3x_pred_vs_gt`

Expected: PASS.

**Step 5: Commit**

```bash
git add debug/inspect_loss_components.py debug/pi3x_train_finite_check.py debug/check_pi3x_pred_mesh_selfcheck.py debug/inspect_pi3x_2d_projection_consistency.py
git commit -m "refactor: remove legacy debug gt dependencies"
```

### Task 3: Update rerun export and object visual checks to the new contract

**Files:**
- Modify: `pi3/visualization/pi3x_rerun_export.py`
- Modify: `debug/test_pi3x_rerun_export.py`
- Modify: `debug/export-dexycb-batch-rrd.py`
- Modify: `debug/test_dexycb_dataset_contract.py`

**Step 1: Write the failing test**

Add assertions that rerun export no longer depends on training-only deprecated GT fields, and that object inspection uses the current `object_multiview` schema only.

**Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_pi3x_rerun_export debug.test_dexycb_dataset_contract`

Expected: FAIL because the exporter still reads legacy fields.

**Step 3: Write minimal implementation**

Keep mesh logging only for visualization if needed, but remove any hard dependency on deprecated training GT keys or legacy mixed `object_multiview` fields.

**Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_pi3x_rerun_export debug.test_dexycb_dataset_contract`

Expected: PASS.

**Step 5: Commit**

```bash
git add pi3/visualization/pi3x_rerun_export.py debug/test_pi3x_rerun_export.py debug/export-dexycb-batch-rrd.py debug/test_dexycb_dataset_contract.py
git commit -m "refactor: align visualization with new batch contract"
```

### Task 4: Clean tensorboard and log-key expectations

**Files:**
- Modify: `debug/test_tensorboard_log_cleanup.py`
- Modify: any trainer logging code touched by the test expectations if needed

**Step 1: Write the failing test**

Assert that `object_2d_loss`, `hand_vertices_loss`, and `hand_beta_loss` are no longer expected logging keys.

**Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_tensorboard_log_cleanup`

Expected: FAIL because the test still expects deprecated tags.

**Step 3: Write minimal implementation**

Update the expected scalar tags and grouped logs to the new contract. Do not broaden logging behavior.

**Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_tensorboard_log_cleanup`

Expected: PASS.

**Step 5: Commit**

```bash
git add debug/test_tensorboard_log_cleanup.py
git commit -m "test: update tensorboard expectations for new loss contract"
```

### Task 5: Remove stale TODO references and summarize module 6

**Files:**
- Modify: `docs/todos/2026-05-13-module1-todo-1.2.md`

**Step 1: Write the failing test**

Not applicable; this is documentation cleanup only.

**Step 2: Update the document**

Move any remaining module 6 completion notes into the TODO markdown and remove stale TODO markers from code comments if any remain.

**Step 3: Verify**

Run:
`rg -n "TODO 1.2|hand_vertices_loss|object_2d_loss|grasped_object_|hand_global_orient_rotmat|hand_pose_rotmat" debug trainers pi3`

Expected: only intentional historical references remain in non-runtime docs or unrelated demos.

**Step 4: Commit**

```bash
git add docs/todos/2026-05-13-module1-todo-1.2.md
git commit -m "docs: record module 6 cleanup"
```
