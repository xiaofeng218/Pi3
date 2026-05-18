# TensorBoard Log Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate TensorBoard output into `train_step`, `train_epoch`, and `val_epoch` scalar-only logs while removing image logging and redundant non-loss tags.

**Architecture:** Keep the existing trainer/loss pipeline intact, but narrow the TensorBoard writer to a scalar whitelist. Step logs will carry per-iteration losses and optimizer parameters, epoch logs will carry aggregated train and validation scalars, and all image visualization will be removed from TensorBoard only. Existing rerun/debug visualization stays untouched.

**Tech Stack:** Python, PyTorch, Hugging Face Accelerate, TensorBoard.

---

### Task 1: Add scalar-only log filtering

**Files:**
- Modify: `trainers/base_trainer_accelerate.py`
- Test: `debug/test_tensorboard_log_cleanup.py`

**Step 1: Write the failing test**

Create a unit test that feeds `log_all()` a mix of loss scalars, training parameters, and image objects, then asserts only allowed scalar tags are sent to TensorBoard and Accelerate.

**Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_tensorboard_log_cleanup -v`

Expected: FAIL because the current logger still emits image entries and non-whitelisted scalars.

**Step 3: Write minimal implementation**

Add a shared scalar filter and remove image emission from the TensorBoard path.

**Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_tensorboard_log_cleanup -v`

Expected: PASS.

### Task 2: Re-tag train and validation summaries

**Files:**
- Modify: `trainers/base_trainer_accelerate.py`
- Modify: `trainers/pi3x_trainer.py`

**Step 1: Write the failing test**

Extend the cleanup test to assert that epoch summaries use `train_epoch/...` and `val_epoch/...`, while per-step summaries use `train_step/...`.

**Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_tensorboard_log_cleanup -v`

Expected: FAIL until the trainer emits the new tag scheme.

**Step 3: Write minimal implementation**

Update the trainer logging calls to emit the new prefixes and keep only loss/parameter scalars.

**Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_tensorboard_log_cleanup -v`

Expected: PASS.
