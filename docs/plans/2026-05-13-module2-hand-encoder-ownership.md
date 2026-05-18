# Module 2 Hand Encoder Ownership Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move hand encoder ownership and hand-query preparation from `Pi3XTrainer` into `Pi3X`, so `Pi3X.forward()` only accepts raw hand inputs.

**Architecture:** `Pi3X` will construct and own the hand encoder, expose a raw-input hand interface (`hand_masks`, `hand_owner_index`, `hand_is_right`), and internally derive `hand_queries` before invoking `HandTokenAdapter`. `Pi3XTrainer` will stop instantiating or running the hand encoder and will only pass raw hand inputs through `model_kwargs`.

**Tech Stack:** PyTorch, Hydra, existing `HaMeREncoder`, existing Pi3X training pipeline

---

### Task 1: Define the new Pi3X hand interface

**Files:**
- Modify: `pi3/models/pi3x.py`
- Modify: `configs/model/pi3x_hand_object.yaml`
- Modify: `configs/overfit.yaml`

**Step 1: Update `Pi3X.__init__` inputs**

Add a model-owned hand encoder config input, for example `hand_encoder_cfg=None`, and store it on the model.

**Step 2: Build the hand encoder inside `Pi3X`**

Add a private `_build_hand_encoder()` helper that:
- returns `None` when no config is provided
- instantiates the encoder with `hydra.utils.instantiate`
- sets it to eval mode
- freezes all encoder parameters

**Step 3: Update config wiring**

Pass the existing hand encoder config through `cfg.model` instead of leaving it trainer-owned.

Run: `rg -n "hand_encoder" configs pi3/models trainers`
Expected: model config owns the hand encoder path, trainer no longer reads a dedicated `hand_encoder` subtree for execution.

**Step 4: Commit**

```bash
git add pi3/models/pi3x.py configs/model/pi3x_hand_object.yaml configs/overfit.yaml
git commit -m "refactor: move hand encoder config ownership into pi3x"
```

### Task 2: Internalize hand query preparation in Pi3X

**Files:**
- Modify: `pi3/models/pi3x.py`
- Test: `debug/test_pi3x_dexycb_hand_integration.py`

**Step 1: Remove `hand_queries` from the public forward interface**

Update `Pi3X.forward()` to accept only:
- `hand_masks`
- `hand_owner_index`
- `hand_is_right`

Delete `hand_queries` from the external signature and from all model callsites after Task 3.

**Step 2: Add `_encode_hands(...)` helper**

Implement a helper in `Pi3X` that:
- returns a normalized empty hand payload when `self.hand_encoder is None`
- returns a normalized empty hand payload when `hand_masks` contains zero rows
- calls `self.hand_encoder(...)` otherwise
- applies `source_index` to reindex `hand_masks` when present
- returns a dict containing:
  - `hand_queries`
  - `hand_masks`
  - `hand_owner_index`
  - `hand_is_right`

**Step 3: Call `_encode_hands(...)` from `Pi3X.forward()`**

Feed the normalized result into `self.hand_token_adapter(...)`.

**Step 4: Update the integration test shape expectations**

Adjust the debug integration test so it constructs raw hand inputs and calls `Pi3X.forward()` without `hand_queries`.

Run: `python -m py_compile pi3/models/pi3x.py debug/test_pi3x_dexycb_hand_integration.py`
Expected: no syntax errors.

**Step 5: Commit**

```bash
git add pi3/models/pi3x.py debug/test_pi3x_dexycb_hand_integration.py
git commit -m "refactor: internalize hand query encoding in pi3x"
```

### Task 3: Remove hand encoder execution from Pi3XTrainer

**Files:**
- Modify: `trainers/pi3x_trainer.py`

**Step 1: Delete trainer-owned hand encoder state**

Remove:
- `self._hand_encoder_cfg`
- `self.hand_encoder`
- `_build_hand_encoder()`

**Step 2: Delete trainer-side hand query execution**

Remove the block in `forward_batch()` that:
- checks `self.hand_encoder`
- creates zero-sized `hand_queries`
- runs `self.hand_encoder(...)`
- handles `source_index`

Keep `_build_hand_inputs()` for now. Module 3 will decide whether it also moves to dataset-side processing.

**Step 3: Update `model_kwargs`**

Stop passing `hand_queries`.
Continue passing:
- `hand_masks`
- `hand_owner_index`
- `hand_is_right`

Run: `rg -n "hand_queries|self.hand_encoder|_build_hand_encoder" trainers/pi3x_trainer.py`
Expected: no remaining runtime use in trainer.

**Step 4: Commit**

```bash
git add trainers/pi3x_trainer.py
git commit -m "refactor: remove trainer-owned hand encoder execution"
```

### Task 4: Verify loading and training-policy compatibility

**Files:**
- Modify: `pi3/models/pi3x.py`
- Modify: `trainers/pi3x_training_policy.py`
- Test: `debug/test_pi3x_trainer_smoke.py`

**Step 1: Confirm hand encoder remains frozen**

Ensure hand encoder parameters stay `requires_grad=False` even after `apply_pi3x_training_policy(...)`.

**Step 2: Confirm new optimizer grouping behavior**

Trainer optimizer grouping should not assume trainer-owned hand encoder parameters exist.

**Step 3: Update smoke coverage if needed**

If the smoke test models assume trainer-owned hand encoder state, update them to align with the new ownership boundary.

Run: `python -m py_compile trainers/pi3x_trainer.py trainers/pi3x_training_policy.py debug/test_pi3x_trainer_smoke.py`
Expected: no syntax errors.

**Step 4: Commit**

```bash
git add pi3/models/pi3x.py trainers/pi3x_training_policy.py debug/test_pi3x_trainer_smoke.py
git commit -m "test: cover pi3x-owned hand encoder flow"
```

### Task 5: Final verification

**Files:**
- Verify only

**Step 1: Run targeted static verification**

Run: `python -m py_compile pi3/models/pi3x.py trainers/pi3x_trainer.py trainers/pi3x_training_policy.py debug/test_pi3x_dexycb_hand_integration.py debug/test_pi3x_trainer_smoke.py`
Expected: all files compile.

**Step 2: Run targeted search checks**

Run: `rg -n "hand_queries" pi3/models/pi3x.py trainers/pi3x_trainer.py debug`
Expected:
- `Pi3X` may still use `hand_queries` internally
- trainer should no longer construct or pass `hand_queries`

**Step 3: Run the focused debug tests if environment is ready**

Run: `python -m unittest debug.test_pi3x_trainer_smoke`
Expected: PASS.

Run: `python -m unittest debug.test_pi3x_dexycb_hand_integration`
Expected: PASS if local DexYCB fixture and HaMeR dependencies are available; otherwise document the exact blocker.

**Step 4: Commit**

```bash
git add docs/plans/2026-05-13-module2-hand-encoder-ownership.md
git commit -m "docs: add module 2 hand encoder ownership plan"
```
