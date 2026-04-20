# Pi3X Data Mirror Mergeback Plan

Date: 2026-04-20
Status: Draft for review
Scope: Moving validated mirror data-pipeline changes back into the main project

## 1. Purpose

This document defines how validated changes from the Pi3X data mirror environment are merged back into the main training repository.

The intent is to avoid:

- reimplementing already tested changes
- mixing debug-only code into production paths
- losing track of which mirrored edits belong in the main project

## 2. Mergeback Principle

Mergeback must be file-driven, not memory-driven.

That means:

- changes are developed and validated in the mirror
- mergeback uses an explicit manifest of source and destination files
- production integration is a controlled sync, not a manual rewrite

## 3. What Is Allowed to Merge Back

Typical mergeback candidates:

- `mirror/datasets/your_dataset.py`
- `mirror/datasets/base/*.py` if contract-preserving fixes were made there
- `mirror/datasets/__init__.py`
- `mirror/pi3/utils/*.py` only when data-path dependencies were updated
- `mirror/configs/data/*.yaml`

Typical non-candidates:

- `debug/*.py`
- `fixtures/*`
- `outputs/*`
- ad hoc notebooks, screenshots, dumps, or exploratory scripts

## 4. Required Mergeback Manifest

Before touching the main project, create a manifest that lists each mirrored file and its destination.

Recommended format:

```text
mirror/datasets/your_dataset.py -> datasets/your_dataset.py [new]
mirror/configs/data/your_dataset.yaml -> configs/data/your_dataset.yaml [new]
mirror/datasets/base/base_dataset.py -> datasets/base/base_dataset.py [modified]
mirror/datasets/__init__.py -> datasets/__init__.py [modified]
```

Each line must indicate:

- mirror source path
- repository destination path
- merge type: `new`, `modified`, or `drop`

## 5. Mergeback Workflow

### Step 1: Freeze the mirror state

Do not merge back from a moving target. Confirm the mirror passes its own checks first.

Minimum required checks:

- index inspection passes
- sample inspection passes
- batch inspection passes
- visualization outputs look sane

### Step 2: Separate production changes from debug tooling

Review the mirror tree and classify each changed file as:

- production-bound
- debug-only
- temporary

Only production-bound files may be included in the mergeback manifest.

### Step 3: Merge new files first

Create new production files before editing shared base files. Typical order:

1. new dataset loader
2. new data config files
3. split files or documented split references if needed

### Step 4: Merge shared-base modifications second

Only after new files are in place, merge any base-pipeline changes:

- `base_dataset.py`
- `transforms.py`
- `utils.py`
- `datasets/__init__.py`

These files are higher risk because they affect existing datasets as well.

### Step 5: Re-run checks in the main project

After mergeback, validate inside the main repository instead of trusting mirror results.

Minimum checks:

- main-project dataloader can be constructed
- one batch can be pulled
- batch fields match expectations
- a small training smoke test runs

## 6. Merge Order Recommendation

Recommended merge order:

1. `datasets/your_dataset.py`
2. `configs/data/your_dataset.yaml`
3. split support files if any
4. `datasets/__init__.py`
5. `datasets/base/*` changes
6. `pi3/utils/*` changes only if required

This reduces the blast radius and makes failures easier to isolate.

## 7. Pre-Mergeback Checklist

All items below must be true:

- mirror changes are stable
- debug-only code is isolated
- no temporary print-based debugging remains in production-bound files
- hardcoded local paths are removed
- new dataset naming is finalized
- split logic is externalized
- sparse-depth strategy is explicitly configured
- pose convention and depth units are documented

## 8. Post-Mergeback Validation

The main project must pass these checks after mergeback.

### 8.1 Loader validation

- `train_loader` builds successfully
- `test_loader` builds successfully if the dataset has a validation split

### 8.2 Batch validation

At least one batch must be pulled successfully and inspected for:

- `img`
- `depthmap`
- `camera_pose`
- `camera_intrinsics`
- `pts3d`
- `valid_mask`
- `sparse_depth`

### 8.3 Smoke training validation

Run a short smoke test:

- one device
- very small iteration count
- no expectation of convergence
- success criterion is no data-path error

## 9. Special Risk Areas

### 9.1 Interface drift

The mirror must not invent a new batch structure. If it does, mergeback is blocked until the mirror is brought back to the production contract.

### 9.2 Dataset label routing

Pi3X uses dataset labels to route some loss behavior. If the new dataset should participate in any special branch, update the relevant dataset-name lists deliberately. If not, leave it out.

### 9.3 Geometry consistency

If crop, resize, depth scaling, or pose conversion changed in the mirror, re-validate geometry after mergeback. Silent geometry mismatch is more dangerous than loader failure.

## 10. Rollback Strategy

If main-project integration fails:

- revert only the production-bound mergeback changes
- do not delete the mirror implementation
- fix and re-validate in the mirror first
- generate a revised mergeback manifest

The mirror remains the source of truth until the mergeback is verified.

## 11. Completion Criteria

Mergeback is complete only when:

- the new dataset works inside the main repository
- the data contract matches the mirror and the trainer expectations
- smoke training passes
- the mergeback manifest reflects the final integrated file set

## 12. Recommendation

Treat mergeback as a controlled promotion step from a validated mirror into the main repository. Avoid manual transcription. Move reviewed files, validate immediately in the main tree, and keep debug tools out of the production path.
