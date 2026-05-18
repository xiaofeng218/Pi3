# Hand Pose PCA Loss Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `hand_pose_loss` supervise the correct MANO hand-joint rotations when dataset `hand_pose_mano` stores PCA coefficients instead of per-joint axis-angle.

**Architecture:** Keep the prediction side unchanged and fix the GT side only. `Pi3XTrainer` will decode dataset MANO pose coefficients into explicit hand-pose rotation matrices using the active MANO layer, then `HandObjectLoss` will prefer those decoded rotmats over naively reshaping `hand_pose_mano[:, 3:]` as axis-angle.

**Tech Stack:** PyTorch, existing Pi3X trainer/loss stack, unittest

---

### Task 1: Lock the expected loss behavior with a failing test

**Files:**
- Modify: `debug/test_hand_object_loss_contract.py`

**Step 1:** Add a test asserting that `HandObjectLoss` uses `gt["hand_pose_rotmat"]` when provided.

**Step 2:** Run the single test and confirm it fails against the current implementation.

### Task 2: Decode GT hand pose coefficients in the trainer

**Files:**
- Modify: `trainers/pi3x_trainer.py`

**Step 1:** Add a helper that decodes `hand_pose_mano` into `hand_global_orient_rotmat` and `hand_pose_rotmat` using the current side-aware MANO layer PCA basis / mean pose.

**Step 2:** Attach the decoded rotmats to `gt` in `forward_batch`.

### Task 3: Consume decoded GT rotmats in the loss

**Files:**
- Modify: `pi3/models/hand_object_loss.py`

**Step 1:** Update `hand_pose_loss` to prefer `gt["hand_pose_rotmat"]`.

**Step 2:** Optionally prefer `gt["hand_global_orient_rotmat"]` for `hand_global_orient_loss` when available for consistency.

### Task 4: Verify

**Files:**
- Test: `debug/test_hand_object_loss_contract.py`
- Test: `debug/test_pi3x_trainer_smoke.py`

**Step 1:** Run the targeted tests.

**Step 2:** If needed, add one trainer smoke test for the PCA decode helper.
