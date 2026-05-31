# ForeHOI Pi3X Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate ForeHOI into the Pi3X training and validation pipeline with a unified dataset contract shared with DexYCB.

**Architecture:** Keep DexYCB and ForeHOI as separate dataset adapters, but normalize both into the same `views` / `scene_inputs` / `gt_metric` / `gt_scale_meta` contract. Move dataset-specific hand-pose decoding and object-scale semantics into dataset-side preprocessing so trainer and loss consume explicit GT tensors instead of inferring dataset meaning.

**Tech Stack:** Python, PyTorch, Hydra, pytest/unittest, NumPy, PIL, OpenCV.

---

### Task 1: Freeze the unified contract with regression tests

**Files:**
- Modify: `debug/test_pi3x_trainer_smoke.py`
- Modify: `debug/test_dexycb_dataset_contract.py`
- Create: `debug/test_forehoi_dataset_contract.py`

**Step 1: Write the failing tests**

Add tests that assert:
- trainer accepts explicit `hand_global_orient_rotmat_gt` / `hand_pose_rotmat_gt`
- trainer accepts explicit `object_scale_canonical_to_scene_metric`
- dataset contract no longer requires `template_vertices`
- ForeHOI dataset contract produces required scene and object multiview fields

**Step 2: Run tests to verify they fail**

Run: `pytest debug/test_pi3x_trainer_smoke.py debug/test_dexycb_dataset_contract.py debug/test_forehoi_dataset_contract.py -v`

Expected: failures due to missing fields / missing dataset implementation.

**Step 3: Implement the minimal test scaffolding**

Create minimal fake ForeHOI fixtures and update existing assertions to the new contract.

**Step 4: Run tests again**

Run the same pytest command and confirm the failures are now only from unimplemented production code.

### Task 2: Refactor trainer and batch utils to consume explicit GT semantics

**Files:**
- Modify: `trainers/pi3x_batch_utils.py`
- Modify: `trainers/pi3x_trainer.py`

**Step 1: Write the failing tests**

Extend trainer smoke tests to assert:
- trainer uses dataset-provided hand rotmat GT directly
- trainer uses `object_scale_canonical_to_scene_metric * scene_scale` for object scale supervision
- trainer no longer expects `object_template_vertices`

**Step 2: Run targeted tests to verify they fail**

Run: `pytest debug/test_pi3x_trainer_smoke.py -v`

**Step 3: Write minimal implementation**

Update batch utils and trainer to:
- aggregate explicit hand rotmat GT
- aggregate explicit object scale GT base
- drop `object_template_vertices` from required training contract

**Step 4: Run tests to verify they pass**

Run: `pytest debug/test_pi3x_trainer_smoke.py -v`

### Task 3: Update DexYCBDataset to emit the unified contract

**Files:**
- Modify: `datasets/dexycb_dataset.py`
- Modify: `debug/test_dexycb_dataset_contract.py`

**Step 1: Write the failing tests**

Add assertions that DexYCB now emits:
- `hand.pose_repr`
- `hand.global_orient_rotmat_gt`
- `hand.pose_rotmat_gt`
- `object.scale_meta`
- `gt_metric.object_scale_canonical_to_scene_metric`
- object multiview without training-time `template_vertices` requirement

**Step 2: Run targeted tests**

Run: `pytest debug/test_dexycb_dataset_contract.py -v`

**Step 3: Write minimal implementation**

Compute hand rotmat GT with the same MANO decoding semantics DexYCB already uses and expose explicit object scale GT semantics while preserving current training behavior.

**Step 4: Run tests**

Run: `pytest debug/test_dexycb_dataset_contract.py -v`

### Task 4: Implement ForeHOIDataset

**Files:**
- Create: `datasets/forehoi_dataset.py`
- Create: `configs/data/forehoi_hand_object.yaml`
- Create: `debug/test_forehoi_dataset_contract.py`

**Step 1: Write the failing tests**

Cover:
- sequence discovery from `pi3x_raw`
- frame loading from RGB/depth/mask/meta
- object multiview loading from `object_multiview_pyrender`
- hand full-axis-angle GT decoding
- explicit object scale GT field construction

**Step 2: Run targeted tests**

Run: `pytest debug/test_forehoi_dataset_contract.py -v`

**Step 3: Write minimal implementation**

Implement dataset loading, multiview caching, hand 2D projection, hand rotmat GT decoding, and object scale metadata assembly.

**Step 4: Run tests**

Run: `pytest debug/test_forehoi_dataset_contract.py -v`

### Task 5: Verify end-to-end compatibility for training entry points

**Files:**
- Modify: `configs/pi3x_hand_object.yaml` only if needed for new data config wiring
- Modify: any affected debug helpers only if tests require them

**Step 1: Write/extend failing coverage**

Add or extend smoke tests to instantiate training batches from both datasets under the unified contract.

**Step 2: Run verification**

Run: `pytest debug/test_pi3x_trainer_smoke.py debug/test_dexycb_dataset_contract.py debug/test_forehoi_dataset_contract.py -v`

**Step 3: Minimal compatibility fixes**

Fix any remaining contract mismatches without expanding scope into visualization refactors.

**Step 4: Run final targeted verification**

Run the same pytest command and confirm all targeted tests pass.

