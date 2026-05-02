# Pi3X Data Mirror Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated Pi3X data mirror environment that preserves the training-facing batch contract while allowing dataset-side debugging, visualization, and future onboarding work.

**Architecture:** Create a self-contained `pi3x_data_mirror/` tree that mirrors the production dataset pipeline files, then add a new dataset adapter skeleton and debug scripts on top. Keep the mirrored pipeline contract-identical to production and keep debug tooling outside the mirrored production path.

**Tech Stack:** Python, PyTorch dataset/dataloader utilities, existing project geometry/cropping helpers, Hydra-style config structure

---

## File Map

### Production-mirror files

- Create: `pi3x_data_mirror/mirror/__init__.py`
- Create: `pi3x_data_mirror/mirror/datasets/__init__.py`
- Create: `pi3x_data_mirror/mirror/datasets/your_dataset.py`
- Create: `pi3x_data_mirror/mirror/datasets/base/base_dataset.py`
- Create: `pi3x_data_mirror/mirror/datasets/base/batched_sampler.py`
- Create: `pi3x_data_mirror/mirror/datasets/base/easy_dataset.py`
- Create: `pi3x_data_mirror/mirror/datasets/base/transforms.py`
- Create: `pi3x_data_mirror/mirror/datasets/base/utils.py`
- Create: `pi3x_data_mirror/mirror/pi3/utils/geometry.py`
- Create: `pi3x_data_mirror/mirror/pi3/utils/cropping.py`
- Create: `pi3x_data_mirror/mirror/configs/data/your_dataset.yaml`
- Create: `pi3x_data_mirror/mirror/configs/data/debug_your_dataset.yaml`

### Debug-only files

- Create: `pi3x_data_mirror/debug/inspect_index.py`
- Create: `pi3x_data_mirror/debug/inspect_sample.py`
- Create: `pi3x_data_mirror/debug/inspect_batch.py`
- Create: `pi3x_data_mirror/debug/visualize_sample.py`

### Documentation / metadata

- Create: `pi3x_data_mirror/README.md`
- Create: `pi3x_data_mirror/MERGEBACK_MANIFEST.md`

## Task 1: Create mirror directory skeleton

**Files:**
- Create: `pi3x_data_mirror/...` directory tree
- Test: directory presence via `find`

- [ ] **Step 1: Create the directory tree**

Required directories:

```text
pi3x_data_mirror/
pi3x_data_mirror/mirror/
pi3x_data_mirror/mirror/datasets/
pi3x_data_mirror/mirror/datasets/base/
pi3x_data_mirror/mirror/pi3/
pi3x_data_mirror/mirror/pi3/utils/
pi3x_data_mirror/mirror/configs/
pi3x_data_mirror/mirror/configs/data/
pi3x_data_mirror/debug/
```

- [ ] **Step 2: Verify the directory tree**

Run:

```bash
find pi3x_data_mirror -maxdepth 4 -type d | sort
```

Expected: the mirror and debug directories all exist.

## Task 2: Copy the minimum production data pipeline into the mirror

**Files:**
- Create: mirrored base pipeline files listed in File Map
- Test: import walk and file presence checks

- [ ] **Step 1: Copy the production dataset base files into the mirror**

Copy these source files as the initial mirror baseline:

```text
datasets/__init__.py
datasets/base/base_dataset.py
datasets/base/batched_sampler.py
datasets/base/easy_dataset.py
datasets/base/transforms.py
datasets/base/utils.py
pi3/utils/geometry.py
pi3/utils/cropping.py
```

- [ ] **Step 2: Add minimal package marker files**

Add empty `__init__.py` files where needed so the mirrored code can import cleanly.

- [ ] **Step 3: Verify copied files are present**

Run:

```bash
find pi3x_data_mirror/mirror -type f | sort
```

Expected: all mirrored files appear under the mirror tree.

## Task 3: Make mirrored imports self-contained

**Files:**
- Modify: mirrored `datasets/__init__.py`
- Modify: mirrored base files with project-root imports
- Test: Python import smoke test

- [ ] **Step 1: Find import paths that still point at the main project tree**

Run:

```bash
rg -n "from datasets|import datasets|from pi3|import pi3|from utils|import utils" pi3x_data_mirror/mirror
```

Expected: a concrete list of import statements that must be remapped or shimmed.

- [ ] **Step 2: Adjust imports so mirrored modules resolve within `pi3x_data_mirror/mirror`**

Target state:

- mirrored dataset code imports mirrored dataset base modules
- mirrored geometry/cropping imports mirrored local dependencies or small copied helpers
- no mirrored file should require importing the production `datasets` package to build a batch

- [ ] **Step 3: Verify importability**

Run:

```bash
python - <<'PY'
import importlib
mods = [
    "pi3x_data_mirror.mirror.datasets",
    "pi3x_data_mirror.mirror.datasets.base.base_dataset",
    "pi3x_data_mirror.mirror.datasets.base.batched_sampler",
]
for name in mods:
    importlib.import_module(name)
print("IMPORT_OK")
PY
```

Expected: `IMPORT_OK`

## Task 4: Add the new dataset skeleton

**Files:**
- Create: `pi3x_data_mirror/mirror/datasets/your_dataset.py`
- Create: `pi3x_data_mirror/mirror/configs/data/your_dataset.yaml`
- Test: import smoke test and config readability

- [ ] **Step 1: Add a minimal `YourDataset` class**

The class should:

- inherit from mirrored `BaseDataset`
- accept `data_root`, `split_file`, and sparse-depth strategy parameters
- expose a placeholder indexing path
- raise a clear `NotImplementedError` only for dataset-specific raw parsing details that are not yet known

- [ ] **Step 2: Add a mirror config**

The config should:

- instantiate `your_dataset.py`
- keep the same train/test dataset structure shape as production configs
- use fixed low-resolution settings in the debug config first

- [ ] **Step 3: Verify the new dataset class imports**

Run:

```bash
python - <<'PY'
from pi3x_data_mirror.mirror.datasets.your_dataset import YourDataset
print(YourDataset.__name__)
PY
```

Expected: `YourDataset`

## Task 5: Add mirror debug tooling

**Files:**
- Create: `pi3x_data_mirror/debug/inspect_index.py`
- Create: `pi3x_data_mirror/debug/inspect_sample.py`
- Create: `pi3x_data_mirror/debug/inspect_batch.py`
- Create: `pi3x_data_mirror/debug/visualize_sample.py`

- [ ] **Step 1: Add index inspection**

The tool must print:

- total sequence count
- frame count statistics
- missing modality counts

- [ ] **Step 2: Add sample inspection**

The tool must report:

- field names
- shapes
- dtypes
- valid-depth ratio
- finite pose check

- [ ] **Step 3: Add batch inspection**

The tool must:

- build the mirrored dataloader
- pull one batch
- print per-view keys and tensor shapes

- [ ] **Step 4: Add sample visualization**

The tool must save output under:

```text
pi3x_data_mirror/outputs/
```

## Task 6: Add README and mergeback manifest scaffold

**Files:**
- Create: `pi3x_data_mirror/README.md`
- Create: `pi3x_data_mirror/MERGEBACK_MANIFEST.md`

- [ ] **Step 1: Document mirror usage**

README must cover:

- environment purpose
- how to run inspect tools
- where the training-facing batch contract lives
- which files are intended for mergeback

- [ ] **Step 2: Create an initial mergeback manifest scaffold**

The manifest must contain headings for:

- new files
- modified mirrored production files
- debug-only files excluded from mergeback

## Task 7: Run a minimum verification sweep

**Files:**
- Test only

- [ ] **Step 1: Verify imports**

Run the import smoke tests from earlier tasks.

- [ ] **Step 2: Verify mirror tree completeness**

Run:

```bash
find pi3x_data_mirror -maxdepth 4 -type f | sort
```

- [ ] **Step 3: Verify no accidental dependency on production dataset package remains in the mirrored runtime path**

Run:

```bash
rg -n "from datasets|import datasets" pi3x_data_mirror/mirror
```

Expected: either no results, or only deliberate compatibility shims that are documented.

- [ ] **Step 4: Record status**

Summarize:

- what is fully working
- what remains placeholder
- what will be needed to onboard the first real dataset instance
