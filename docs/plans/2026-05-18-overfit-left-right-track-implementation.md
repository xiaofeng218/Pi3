# Overfit Left Right Track Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make overfit training use two explicitly configured DexYCB tracks, one left-hand and one right-hand.

**Architecture:** Extend `DexYCBDataset` with a small explicit track resolver that filters the split-scoped track list by `subject + sequence + camera`. Wire the overfit config to provide one left and one right selector, and validate the behavior with a dataset contract test and config smoke test.

**Tech Stack:** Hydra configs, Python dataset code, unittest-style debug contract tests.

---

### Task 1: Add a failing dataset contract test for explicit left/right selection

**Files:**
- Modify: `debug/test_dexycb_dataset_contract.py`

**Step 1: Write the failing test**

Add a fixture that includes one left-hand sequence and one right-hand sequence, then instantiate `DexYCBDataset(selected_tracks=...)` and assert the remaining tracks are exactly those two.

**Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_dexycb_dataset_contract`

Expected: FAIL because `DexYCBDataset` does not yet understand `selected_tracks`.

**Step 3: Write minimal implementation**

Implement dataset-side selector parsing and filtering.

**Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_dexycb_dataset_contract`

Expected: PASS

### Task 2: Add overfit config assertions

**Files:**
- Modify: `debug/test_pi3x_config_smoke.py`
- Modify: `configs/overfit.yaml`

**Step 1: Write the failing config assertion**

Assert that `cfg.train_dataset.selected_tracks.left/right` exist and that `max_tracks` is no longer the mechanism.

**Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_pi3x_config_smoke`

Expected: FAIL because config still uses `max_tracks: 1`.

**Step 3: Write minimal implementation**

Replace the overfit track truncation config with explicit left/right selectors.

**Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_pi3x_config_smoke`

Expected: PASS

### Task 3: Verify both targeted tests stay green

**Files:**
- No code changes required unless a regression appears

**Step 1: Run targeted tests**

Run:

```bash
python -m unittest debug.test_dexycb_dataset_contract
python -m unittest debug.test_pi3x_config_smoke
```

Expected: PASS

**Step 2: Commit**

```bash
git add debug/test_dexycb_dataset_contract.py debug/test_pi3x_config_smoke.py configs/overfit.yaml datasets/dexycb_dataset.py docs/plans/2026-05-18-overfit-left-right-track-design.md docs/plans/2026-05-18-overfit-left-right-track-implementation.md
git commit -m "feat: select explicit left and right overfit tracks"
```
