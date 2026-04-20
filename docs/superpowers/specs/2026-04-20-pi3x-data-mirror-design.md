# Pi3X Data Mirror Environment Design

Date: 2026-04-20
Status: Draft for review
Scope: Isolated data pipeline mirror for new dataset onboarding and debugging

## 1. Purpose

This document defines a mirror data environment for Pi3X. The mirror is intended to:

- isolate dataset-side experimentation from the main training codebase
- preserve the same batch interface consumed by the training stack
- support direct visualization and inspection of samples and batches
- allow new dataset onboarding to be developed and validated before mergeback
- make validated mirror changes easy to move back into the main project

The mirror is not a replacement trainer. It mirrors the data path from raw files to training-ready batches.

## 2. Goals

- Preserve the main project data contract end-to-end.
- Keep dataset-side code as close as possible to the original implementation style.
- Provide a fast local loop for inspecting index building, sample generation, and batched outputs.
- Make `your_dataset.py` the main adaptation surface for new data.
- Keep mergeback work file-based rather than rewrite-based.

## 3. Non-Goals

- Refactoring the trainer, model, optimizer, or scheduler stack.
- Redesigning the main dataset abstractions.
- Introducing a new universal intermediate format for all future datasets.
- Improving code style in unrelated parts of the repository.

## 4. Output Contract

The mirror environment must produce the same training-facing structures as the main project.

### 4.1 Dataset sample contract

Each dataset sample must remain:

```python
list[view_dict]
```

Each `view_dict` must contain at least:

- `img`
- `depthmap`
- `camera_pose`
- `camera_intrinsics`
- `dataset`
- `label`
- `instance`
- `sparse_depth`

The mirrored base pipeline must continue to derive:

- `idx`
- `true_shape`
- `z_far`
- `pts3d`
- `valid_mask`
- `normal`

### 4.2 Dataloader batch contract

The collated batch must remain:

```python
list[batched_view_dict]
```

This must stay compatible with the current training code path:

- trainer stacks `view["img"]` into `(B, N, C, H, W)`
- Pi3X loss stacks `pts3d`, `valid_mask`, `camera_pose`, `camera_intrinsics`, and `sparse_depth`

No debug tool may change this output contract.

## 5. Architecture

The mirror is split into two layers:

1. Mirror pipeline code
2. Debug and visualization tools

The mirror pipeline code is intended for later mergeback. The debug layer is not.

### 5.1 Proposed layout

```text
docs/superpowers/specs/
  2026-04-20-pi3x-data-mirror-design.md
  2026-04-20-pi3x-data-mirror-mergeback.md

pi3x_data_mirror/
  mirror/
    datasets/
      __init__.py
      scannet_dataset.py
      tartanair_dataset.py
      co3dv2_dataset.py
      your_dataset.py
      base/
        base_dataset.py
        batched_sampler.py
        easy_dataset.py
        transforms.py
        utils.py
      sample_utils/
        ...
    pi3/
      utils/
        geometry.py
        cropping.py
        basic.py
        alignment.py
    configs/
      data/
        your_dataset.yaml
        debug_your_dataset.yaml
      train/
        debug_local.yaml
  debug/
    inspect_index.py
    inspect_sample.py
    inspect_batch.py
    visualize_sample.py
    visualize_batch.py
  fixtures/
    mini_dataset/
  outputs/
```

## 6. Copy Strategy

### 6.1 Files mirrored with minimal changes

These files define the core data contract and should be copied first with minimal edits:

- `datasets/base/base_dataset.py`
- `datasets/base/batched_sampler.py`
- `datasets/base/easy_dataset.py`
- `datasets/base/transforms.py`
- `datasets/base/utils.py`
- `datasets/__init__.py`
- `pi3/utils/geometry.py`
- `pi3/utils/cropping.py`
- any small utility modules strictly required by the above

### 6.2 Files copied mainly as references

- `datasets/scannet_dataset.py`
- `datasets/tartanair_dataset.py`
- `datasets/co3dv2_dataset.py`

These are baseline loader references for sampling strategy and field assembly.

### 6.3 Files newly authored in the mirror

- `mirror/datasets/your_dataset.py`
- `mirror/configs/data/your_dataset.yaml`
- `mirror/configs/data/debug_your_dataset.yaml`
- `debug/*.py`

## 7. New Dataset Adapter Design

`your_dataset.py` is the main adaptation file for the new dataset.

Its responsibilities are limited to:

- building a sequence index
- loading frame-level RGB, depth, intrinsics, and pose
- sampling multi-frame views
- assembling standardized `view_dict` outputs

It should not include:

- visualization logic
- one-off debug print flows
- unrelated data cleaning pipelines
- split generation rules hardcoded in the loader
- experimental trainer-specific hacks

Recommended internal helper boundaries:

- `build_index(data_root, split_file)`
- `load_frame_meta(sequence_id, frame_id)`
- `load_rgb(meta)`
- `load_depth(meta)`
- `load_intrinsics(meta)`
- `load_pose(meta)`
- `make_sparse_depth(depthmap, strategy, rng)`

## 8. Split Strategy

Train and validation splits should be externalized. The loader should consume splits, not define them.

Recommended options:

- `splits/train.txt` and `splits/valid.txt`
- a single split JSON manifest

Each split entry should identify a sequence, not an individual frame, unless the source dataset requires frame-level exclusion.

## 9. Sparse Depth Strategy

Pi3X expects `sparse_depth`-derived supervision. The mirror must expose this as a configurable strategy.

Supported modes:

- `all_valid`
- `grid_subsample`
- `random_subsample`
- `disabled`

Recommended defaults:

- debug path: `all_valid`
- real fine-tuning path: `grid_subsample`

This keeps the pipeline easy to validate first, then closer to intended sparse supervision behavior later.

## 10. Sampling Strategy

The new loader must respect dynamic `frame_num` from the base pipeline. It must not hardcode the view count.

Recommended initial strategy:

- local window sampling around a randomly selected anchor frame
- optional global random sampling branch for long sequences
- support for replacement when a sequence is shorter than requested frame count

This keeps behavior aligned with the current project pattern and avoids a dataset-specific special case.

## 11. Validation and Debug Tooling

Debug tools must operate on the mirrored standard outputs rather than on raw source files.

Required tools:

- `inspect_index.py`
  - sequence counts
  - frame count distribution
  - missing modality checks
- `inspect_sample.py`
  - field presence
  - shapes and dtypes
  - valid depth ratio
  - finite pose check
- `inspect_batch.py`
  - actual dataloader construction
  - one-batch contract validation
- `visualize_sample.py`
  - multi-view RGB, depth, valid mask, sparse depth
- `visualize_batch.py`
  - batched sampling inspection across multiple examples

## 12. Development Phases

### Phase 1: Mirror bootstrap

- create the mirror directory
- copy the base pipeline files
- make imports self-contained

Exit criteria:

- mirror package imports cleanly

### Phase 2: New dataset indexing

- implement index building
- add split handling
- verify raw file integrity

Exit criteria:

- index inspection script succeeds

### Phase 3: Sample generation

- implement `your_dataset.py`
- produce valid `view_dict` samples

Exit criteria:

- sample inspection succeeds

### Phase 4: Batch generation

- wire mirrored dataloader creation
- validate collated outputs

Exit criteria:

- batch inspection succeeds

### Phase 5: Debug and mergeback readiness

- add visualization tools
- produce mergeback manifest
- confirm output contract matches main project expectations

Exit criteria:

- documented and repeatable mergeback path exists

## 13. Acceptance Criteria

The mirror environment is complete only if all items below are true:

- one new dataset can be indexed successfully
- one sample can be generated without contract violations
- one training-ready batch can be produced from the mirror dataloader
- batch structure matches the main project contract
- RGB, depth, valid mask, sparse depth, and pose sanity can be visualized
- no debug-only behavior is embedded into the batch contract

## 14. Risks

- import drift between mirror code and the main project
- accidental interface drift caused by debug conveniences
- incorrect pose convention or depth units
- intrinsics not updated consistently after crop and resize
- `dataset` label not aligned with Pi3X loss routing behavior

## 15. Recommendation

Implement the mirror as a controlled copy of the existing data pipeline plus a new dataset adapter and separate debug tools. Avoid redesigning the pipeline at this stage. The shortest path to safe onboarding is a mirror that behaves like production code while remaining isolated from the production repository paths.
