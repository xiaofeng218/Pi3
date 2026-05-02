# Pi3X Object Dual-Stream Implementation Plan

> Updated on 2026-04-24 to reflect the current `ho_decoder[i]` implementation direction. This version supersedes the earlier masked-routing plan.

**Goal:** Extend `Pi3X` with a shared-weight object canonical multiview stream, scene-side object query tokens, per-layer HO query blocks, and an object pose head while preserving the existing scene and hand paths.

**Current architecture direction:**
- `scene` and `object canonical multiview` streams run through the existing shared `decoder[i]`
- `hand/object query` tokens run through a separate per-layer `ho_decoder[i]`
- `i % 2 == 0`:
  - `ho_decoder[i]` updates the merged `hand + object` query line with `self-attn + cross-attn(query <- scene hidden)`
- `i % 2 == 1`:
  - `hand` runs through `ho_decoder[i]` with `self-attn` only
  - `object` runs through `ho_decoder[i]` with `self-attn + cross-attn(query <- object hidden_omv)`
- invalid `hand/object` slots use gated writeback and remain stable

**Tech Stack:** PyTorch, existing Pi3X modules under `pi3/models`, Python `unittest`, debug-oriented tests under `debug/`, DexYCB object payloads emitted by `datasets/dexycb_dataset.py`.

---

## Completed Work

### Task 1: Object query and pose module tests
- [x] Added `debug/test_pi3x_object_modules.py`
- [x] Added coverage for `ObjectQueryAdapter`
- [x] Added coverage for `ObjectPoseHead`

### Task 2: Object query and pose modules
- [x] Implemented `pi3/models/layers/object_query_adapter.py`
- [x] Implemented `pi3/models/layers/object_pose_head.py`
- [x] Verified the object module test file passes

### Task 3: Object dual-stream integration test
- [x] Added `debug/test_pi3x_object_dual_stream.py`
- [x] Kept the test aligned with the current `ho_decoder[i]` route
- [x] Verified canonical stream changes valid object-query outputs while invalid views remain stable

### Task 4: Pi3X object dual-stream path
- [x] Extended `Pi3X.forward(..., object_multiview=...)`
- [x] Added `decode_prepare(...)`
- [x] Refactored `decode(...)` into scene/object/query routing
- [x] Added `HOBlockRope`
- [x] Added `self.ho_decoder = nn.ModuleList([...])`
- [x] Routed hand/object query tokens entirely through `ho_decoder[i]`
- [x] Added hand/object valid-gated writeback
- [x] Emitted object pose predictions from object-query penultimate/final features

---

## Remaining Work

### Task 5: Warm-start `ho_decoder[i]` from `decoder[i]`

**Why:** The current `ho_decoder[i]` blocks are structurally correct but still use default initialization. The next step is to initialize them from the corresponding `decoder[i]` blocks so they inherit the original Pi3X representation.

**Files:**
- Modify: `pi3/models/layers/block.py`
- Modify: `pi3/models/pi3x.py`
- Create or modify: `debug/test_pi3x_ho_decoder_init.py`

- [x] **Step 1: Add a block-level warm-start helper**

Add a helper that maps `BlockRope -> HOBlockRope`:

```python
def init_ho_block_from_decoder_block(ho_blk: HOBlockRope, blk: BlockRope) -> None:
    ...
```

Expected mapping:
- `norm1`, `attn`, `norm3`, `mlp`, `ls1`, `ls2` copied directly where shapes match
- `cross_attn.q/k/v/proj` initialized from the corresponding `blk.attn.qkv/proj`
- `norm2` / `norm_y` initialized from decoder norms

- [x] **Step 2: Initialize `self.ho_decoder[i]` inside `Pi3X.__init__`**

After both `decoder` and `ho_decoder` are created:

```python
for blk, ho_blk in zip(self.decoder, self.ho_decoder):
    init_ho_block_from_decoder_block(ho_blk, blk)
```

- [x] **Step 3: Add a focused init test**

Test should verify:
- copied weights are numerically equal where expected
- `ho_decoder[i]` cross-attn weights are no longer default-random relative to decoder projections

- [x] **Step 4: Run the init test and object-path regression suite**

Run:

```bash
python -m unittest \
  debug.test_pi3x_ho_decoder_init \
  debug.test_pi3x_object_modules \
  debug.test_pi3x_object_dual_stream -v
```

- [ ] **Step 5: Commit the warm-start implementation**

```bash
git add pi3/models/layers/block.py pi3/models/pi3x.py debug/test_pi3x_ho_decoder_init.py
git commit -m "feat: warm start Pi3X ho decoder from scene decoder"
```

### Task 6: Add DexYCB object-path contract and integration coverage

**Files:**
- Modify: `debug/test_dexycb_dataset_contract.py`
- Create: `debug/test_pi3x_dexycb_object_integration.py`

- [x] **Step 1: Add dataset contract assertions for object payloads**

Verify:
- `object_multiview` exists
- `grasped_object_mask` matches the current view image/depth shape
- `grasped_object_valid` is boolean-like
- canonical object multiview RGB shape is `(V, 3, 224, 224)`

- [x] **Step 2: Add a DexYCB object dual-stream integration test**

Test should:
- construct a real or fixture-backed DexYCB batch
- pass `imgs` and `object_multiview` into `Pi3X.forward`
- assert:
  - `pred_object_rot6d`
  - `pred_object_trans`
  - `pred_object_scale`
  - `object_valid`

- [x] **Step 3: Run the object integration test**

Run:

```bash
python -m unittest debug.test_pi3x_dexycb_object_integration -v
```

- [ ] **Step 4: Commit DexYCB object coverage**

```bash
git add debug/test_dexycb_dataset_contract.py debug/test_pi3x_dexycb_object_integration.py
git commit -m "test: cover DexYCB object dual-stream path"
```

### Task 7: Full verification and cleanup

**Files:**
- Modify: none unless cleanup is required

- [x] **Step 1: Run the focused object-path suite**

Run:

```bash
python -m unittest \
  debug.test_pi3x_object_modules \
  debug.test_pi3x_object_dual_stream \
  debug.test_pi3x_ho_decoder_init \
  debug.test_dexycb_dataset_contract \
  debug.test_pi3x_dexycb_object_integration -v
```

- [x] **Step 2: Run hand regression coverage**

Run:

```bash
python -m unittest debug.test_pi3x_hand_modules -v
```

- [x] **Step 3: Remove obsolete masked-routing artifacts**

This cleanup is part of the current plan revision:
- `pi3/models/layers/dual_stream_routing.py`
- `debug/test_pi3x_dual_stream_routing.py`
- plan references to masked scene-routing as the primary architecture

Note:
- generic boolean mask support already living in `attention.py` may remain if still useful as shared infrastructure
- it is no longer considered part of the primary Pi3X object dual-stream route

- [x] **Step 4: Confirm clean object-path implementation state**

Run:

```bash
git status --short
```

Observed:
- object-path edits are present as intended
- the repo still contains unrelated pre-existing dirty/untracked files outside this change scope
- no temporary masked-routing hooks remain in the active Pi3X object path

- [ ] **Step 5: Commit verification/cleanup state**

```bash
git add docs/superpowers/plans/2026-04-24-pi3x-object-dual-stream-implementation.md
git commit -m "chore: update Pi3X object dual-stream implementation plan"
```

---

## Compatibility Notes

This revised plan is compatible with earlier completed work:

- `ObjectQueryAdapter` and `ObjectPoseHead` remain unchanged in role
- existing object dual-stream tests remain valid, but now target `ho_decoder[i]`
- earlier masked-routing utilities are no longer part of the main path and have been removed from the plan
- the remaining implementation work is incremental:
  - add `ho_decoder` warm start
  - add DexYCB object integration coverage
  - run full verification and cleanup

## Self-Review

- The plan now matches the current code direction:
  - `decoder[i]` for scene/object streams
  - `ho_decoder[i]` for hand/object queries
- The old masked-routing route is no longer described as the primary architecture
- Remaining work is focused on initialization, dataset integration, and verification rather than another architecture rewrite
