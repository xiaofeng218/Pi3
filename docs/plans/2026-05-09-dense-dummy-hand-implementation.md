# Dense Dummy Hand Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert the hand branch to an always-on dense-slot path with a dummy slot for no-hand batches so dual-GPU DDP no longer sees rank-dependent unused hand parameters.

**Architecture:** Keep dynamic per-batch hand slot counts, but force one dense dummy slot when a batch has no hands. Run dense hand slots through `decode()` and `hand_mano_head` on every step, then mask invalid slots in the dense hand loss path so no-hand batches contribute zero supervision while preserving graph participation.

**Tech Stack:** PyTorch, DDP via Accelerate, Hydra configs, `unittest`

---

### Task 1: Lock down current adapter behavior with tests

**Files:**
- Modify: `debug/test_pi3x_object_modules.py`
- Modify: `pi3/models/layers/hand_token_adapter.py`

**Step 1: Write the failing test**

Add a test that calls `HandTokenAdapter.forward()` with zero-row `hand_queries` and asserts:

- `dense_tokens` is not `None`
- `dense_tokens.shape[2] == 1`
- `dense_valid_mask.shape == (B, N, 1)`
- `dense_valid_mask.any() == False`
- `num_hand_tokens == 1`

**Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_pi3x_object_modules -q`

Expected: FAIL because empty hand input currently returns `dense_tokens=None` and `num_hand_tokens=0`.

**Step 3: Write minimal implementation**

Update `HandTokenAdapter.forward()` so empty input returns one dense dummy slot built from `empty_hand_token`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_pi3x_object_modules -q`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_pi3x_object_modules.py pi3/models/layers/hand_token_adapter.py
git commit -m "test: cover dummy dense hand slot adapter behavior"
```

### Task 2: Keep empty hand batches on the model path

**Files:**
- Modify: `debug/test_pi3x_trainer_smoke.py`
- Modify: `trainers/pi3x_trainer.py`

**Step 1: Write the failing test**

Add a trainer smoke test asserting that empty hand encoder outputs are no longer collapsed to `None` before `self.model(**model_kwargs)`.

**Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_pi3x_trainer_smoke.Pi3XTrainerSmokeTests -q`

Expected: FAIL because the trainer currently converts empty hand encoder outputs to `None`.

**Step 3: Write minimal implementation**

Change `Pi3XTrainer.forward_batch()` to preserve the dense dummy hand path instead of dropping it.

**Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_pi3x_trainer_smoke.Pi3XTrainerSmokeTests -q`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_pi3x_trainer_smoke.py trainers/pi3x_trainer.py
git commit -m "test: keep empty hand batches on dense model path"
```

### Task 3: Convert the hand head path from sparse to dense

**Files:**
- Modify: `debug/test_pi3x_trainer_smoke.py`
- Modify: `pi3/models/pi3x.py`

**Step 1: Write the failing test**

Add a model/trainer test covering a no-hand batch and asserting:

- `pred_hand_transl` exists
- dense hand outputs include one slot
- outputs remain present when `hand_valid_mask` is all `False`

**Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_pi3x_trainer_smoke -q`

Expected: FAIL because the current model omits `pred_hand_*` when there are no real sparse hand features.

**Step 3: Write minimal implementation**

Update `Pi3X.forward()` to:

- flatten dense hand slots to `B*N*M`
- run every slot through `hand_mano_head`
- reshape predictions back to dense tensors
- stop requiring sparse real-hand collection to produce hand predictions

**Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_pi3x_trainer_smoke -q`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_pi3x_trainer_smoke.py pi3/models/pi3x.py
git commit -m "feat: run dense dummy hand slots through hand head"
```

### Task 4: Densify hand GT and masked loss behavior

**Files:**
- Modify: `debug/test_pi3x_trainer_smoke.py`
- Modify: `debug/test_pi3x_object_modules.py`
- Modify: `pi3/models/hand_object_loss.py`
- Modify: `trainers/pi3x_trainer.py`

**Step 1: Write the failing test**

Add tests that assert:

- dense hand GT tensors align with dense prediction slots
- fully invalid dense hand batches produce zero hand loss values
- hand loss keys still exist in `details`

**Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_pi3x_trainer_smoke debug.test_pi3x_object_modules -q`

Expected: FAIL because current GT/loss code is sparse-first and prediction-presence dependent.

**Step 3: Write minimal implementation**

Update trainer GT preparation and `HandObjectLoss.forward()` so hand loss always consumes dense tensors plus a dense `hand_valid_mask`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_pi3x_trainer_smoke debug.test_pi3x_object_modules -q`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_pi3x_trainer_smoke.py debug/test_pi3x_object_modules.py pi3/models/hand_object_loss.py trainers/pi3x_trainer.py
git commit -m "feat: add dense masked hand supervision path"
```

### Task 5: Re-verify dual-GPU behavior and logging

**Files:**
- Modify if needed: `trainers/base_trainer_accelerate.py`
- Reuse: `outputs/` debug logs from local runs

**Step 1: Run focused tests**

Run:

```bash
python -m unittest debug.test_base_trainer_accelerate_contract -q
python -m unittest debug.test_pi3x_trainer_smoke -q
python -m unittest debug.test_pi3x_object_modules -q
```

Expected: PASS

**Step 2: Run dual-GPU reproduction command**

Run:

```bash
source asset_registry/env.sh
PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 scripts/train_pi3x.py \
name=stall_debug_dense_hand \
log.output_dir=outputs/stall_debug_dense_hand \
log.ckpt_dir=outputs/stall_debug_dense_hand/ckpts \
train.num_epoch=1 train.iters_per_epoch=90 \
+test.validate_before_train=false \
log.step_timing=true log.step_timing_interval=1 log.step_timing_log_all_ranks=true \
train.num_workers=0 test.num_workers=0 vis.interval=0
```

Expected: run passes the previous `global_step=87` hang point without stalling due to rank-dependent hand branch usage.

**Step 3: Inspect summaries**

Run:

```bash
rg -n "global_step=87|global_step=88" outputs/stall_debug_dense_hand/step_summary_rank*.log
```

Expected: at least one no-hand/dummy-hand step still produces dense hand outputs without DDP hang.

**Step 4: Commit**

```bash
git add trainers/base_trainer_accelerate.py
git commit -m "chore: verify dense dummy hand branch under dual gpu"
```
