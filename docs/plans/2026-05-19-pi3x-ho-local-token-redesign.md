# Pi3X HO Local Token Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the old single-token HO branch with dense `(B, N)` local hand/object token sequences while keeping the scene branch and object-multiview branch intact.

**Architecture:** The scene branch remains the global backbone. Hand/object branches switch to crop-based local token encoding, dense `(B, N)` token organization, and a redesigned HO decoder that runs in parallel with scene/object-multiview decoding. Prediction moves from single-token regression to pooled local-token heads. This redesign removes the old sparse hand pathway and the `hand_owner_index` dependency.

**Tech Stack:** PyTorch, DINOv2, HaMeR, Hydra, existing Pi3X trainer/loss stack

---

### Task 1: Lock Down Crop Geometry Helpers

**Files:**
- Create: `debug/test_local_crop_geometry.py`
- Modify: `pi3/models/hamer/encoder.py`
- Modify: `pi3/models/layers/hand_token_adapter.py`
- Modify: `pi3/models/layers/object_query_adapter.py`

**Step 1: Write the failing tests**

Add tests for:
- bbox extraction from mask
- expanded crop box clipping
- left/right canonicalization behavior
- `(256, 192)` and `(224, 168)` resize paths preserving token-count-compatible grids

**Step 2: Run test to verify it fails**

Run: `pytest debug/test_local_crop_geometry.py -v`

Expected: failing or missing helper coverage

**Step 3: Extract reusable crop helpers**

Refactor crop helpers into a shared implementation used by:
- `HaMeREncoder`
- `HandTokenAdapter`
- `ObjectQueryAdapter`

Keep one crop-box definition for both hand branches.

**Step 4: Run test to verify it passes**

Run: `pytest debug/test_local_crop_geometry.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_local_crop_geometry.py pi3/models/hamer/encoder.py pi3/models/layers/hand_token_adapter.py pi3/models/layers/object_query_adapter.py
git commit -m "refactor: unify local crop geometry helpers"
```

### Task 2: Convert HaMeR Encoder Output From Query To Context Tokens

**Files:**
- Create: `debug/test_hamer_context_tokens.py`
- Modify: `pi3/models/hamer/backbone_query.py`
- Modify: `pi3/models/hamer/encoder.py`

**Step 1: Write the failing tests**

Add tests for:
- `HaMeRBackbone` returning context tokens instead of query token
- `HaMeREncoder` returning `(B, N)`-aligned dense hand outputs
- betas still emitted correctly
- `hand_owner_index` no longer required by the hand branch interface

**Step 2: Run test to verify it fails**

Run: `pytest debug/test_hamer_context_tokens.py -v`

Expected: FAIL because old query output contract is still active

**Step 3: Write minimal implementation**

Change:
- `HaMeRBackbone.forward()` to expose context token sequence plus betas
- `HaMeREncoder.forward()` to produce dense `(B, N, T_hand, C)` token layout and `(B, N)` valid mask semantics

**Step 4: Run test to verify it passes**

Run: `pytest debug/test_hamer_context_tokens.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_hamer_context_tokens.py pi3/models/hamer/backbone_query.py pi3/models/hamer/encoder.py
git commit -m "feat: expose hamer context tokens for hand branch"
```

### Task 3: Rewrite HandTokenAdapter And ObjectQueryAdapter As Crop Token Encoders

**Files:**
- Create: `debug/test_local_token_adapters.py`
- Modify: `pi3/models/layers/hand_token_adapter.py`
- Modify: `pi3/models/layers/object_query_adapter.py`
- Modify: `pi3/models/pi3x.py`

**Step 1: Write the failing tests**

Add tests for:
- hand adapter outputs DINO crop token sequences instead of pooled single tokens
- object adapter outputs DINO crop token sequences instead of pooled single token
- invalid views are filled with empty token sequences
- outputs are dense `(B, N, T, C)` with `(B, N)` valid masks

**Step 2: Run test to verify it fails**

Run: `pytest debug/test_local_token_adapters.py -v`

Expected: FAIL because adapters still implement pooled-token behavior

**Step 3: Write minimal implementation**

Replace internal adapter logic with:
- crop from RGB
- resize
- encode with external scene DINOv2
- emit dense token sequences
- side embedding on hand path
- empty token sequences for invalid views

**Step 4: Run test to verify it passes**

Run: `pytest debug/test_local_token_adapters.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_local_token_adapters.py pi3/models/layers/hand_token_adapter.py pi3/models/layers/object_query_adapter.py pi3/models/pi3x.py
git commit -m "feat: replace ho adapters with local crop token encoders"
```

### Task 4: Densify Hand/Object Branch Interfaces In Pi3X Forward

**Files:**
- Create: `debug/test_pi3x_dense_local_tokens.py`
- Modify: `pi3/models/pi3x.py`

**Step 1: Write the failing tests**

Add tests for:
- dense `(B, N)` hand token contract
- dense `(B, N)` object token contract
- removal of sparse hand gather/scatter path and `hand_owner_index` dependency
- preservation of scene and object-multiview inputs

**Step 2: Run test to verify it fails**

Run: `pytest debug/test_pi3x_dense_local_tokens.py -v`

Expected: FAIL because current forward path still expects sparse hand organization

**Step 3: Write minimal implementation**

Update `Pi3X.forward()` to:
- stop using sparse hand token gathering
- consume dense hand/object token outputs
- retain `object_multiview` as a separate modality

**Step 4: Run test to verify it passes**

Run: `pytest debug/test_pi3x_dense_local_tokens.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_pi3x_dense_local_tokens.py pi3/models/pi3x.py
git commit -m "refactor: convert pi3x hand and object branches to dense local tokens"
```

### Task 5: Redesign HO Decoder Block Execution

**Files:**
- Create: `debug/test_ho_decoder_local_token_schedule.py`
- Modify: `pi3/models/pi3x.py`
- Modify: `pi3/models/layers/block.py`

**Step 1: Write the failing tests**

Add tests for:
- even-layer per-view hand/object self-attention
- even-layer scene cross-attention with zero-initialized residual scaling
- odd-layer cross-view self-attention
- odd-layer object-to-object-multiview cross-attention
- preservation of scene decoder and object-multiview decoder layer progression

**Step 2: Run test to verify it fails**

Run: `pytest debug/test_ho_decoder_local_token_schedule.py -v`

Expected: FAIL because current HO decoder still uses single-token query behavior

**Step 3: Write minimal implementation**

Refactor HO execution to:
- keep 36-layer schedule
- keep scene/object-multiview decoding active each layer
- separate hand/object token updates
- add new cross-attn modules with zero-init residual scalars

**Step 4: Run test to verify it passes**

Run: `pytest debug/test_ho_decoder_local_token_schedule.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_ho_decoder_local_token_schedule.py pi3/models/pi3x.py pi3/models/layers/block.py
git commit -m "feat: redesign ho decoder for dense local token attention"
```

### Task 6: Add New Pooling Heads And Prediction Heads

**Files:**
- Create: `debug/test_local_token_prediction_heads.py`
- Modify: `pi3/models/pi3x.py`
- Modify: `pi3/models/hamer/hand_mano_head.py`
- Modify: `pi3/models/layers/object_pose_head.py`
- Create: `pi3/models/layers/hand_global_head.py`
- Create: `pi3/models/layers/token_pool_head.py`

**Step 1: Write the failing tests**

Add tests for:
- object token pooling -> object pose prediction
- shared pooled hand feature -> global head + hand pose head + MANO wrapper
- prediction key contract matching dense output flow
- hand betas still sourced from HaMeR
- `hand_pose_head` preserving the old rotmat-compatible output contract
- `HandMANOHead` acting only as a MANO parameter integration/geometry module

**Step 2: Run test to verify it fails**

Run: `pytest debug/test_local_token_prediction_heads.py -v`

Expected: FAIL because current prediction path still assumes single-token heads

**Step 3: Write minimal implementation**

Implement:
- token pooling module
- `hand_global_head`
- `hand_pose_head`
- `HandMANOHead` as pure MANO forward/wrapper
- pooled hand/object prediction flow
- shared pooled hand feature contract

**Step 4: Run test to verify it passes**

Run: `pytest debug/test_local_token_prediction_heads.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_local_token_prediction_heads.py pi3/models/pi3x.py pi3/models/hamer/hand_mano_head.py pi3/models/layers/object_pose_head.py pi3/models/layers/hand_global_head.py pi3/models/layers/token_pool_head.py
git commit -m "feat: add pooled local-token prediction heads"
```

### Task 7: Replace Sparse Hand Loss Flow With Dense Masked Supervision

**Files:**
- Create: `debug/test_dense_hand_object_loss.py`
- Modify: `pi3/models/hand_object_loss.py`
- Modify: `trainers/pi3x_trainer.py`
- Modify: `trainers/pi3x_batch_utils.py`

**Step 1: Write the failing tests**

Add tests for:
- dense `(B, N)` hand GT construction
- dense `(B, N)` hand/object prediction supervision
- valid-mask loss filtering
- removal of sparse hand owner-index supervision path

**Step 2: Run test to verify it fails**

Run: `pytest debug/test_dense_hand_object_loss.py -v`

Expected: FAIL because trainer/loss still depend on sparse hand alignment

**Step 3: Write minimal implementation**

Update:
- GT construction to keep dense semantics
- trainer to stop sparsifying hand GT
- loss functions to use dense masked supervision

**Step 4: Run test to verify it passes**

Run: `pytest debug/test_dense_hand_object_loss.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_dense_hand_object_loss.py pi3/models/hand_object_loss.py trainers/pi3x_trainer.py trainers/pi3x_batch_utils.py
git commit -m "refactor: migrate hand object loss to dense masked supervision"
```

### Task 8: Add Initialization And Stage-1 Training Policy

**Files:**
- Create: `debug/test_ho_decoder_initialization_policy.py`
- Modify: `pi3/models/pi3x.py`
- Modify: `trainers/pi3x_training_policy.py`

**Step 1: Write the failing tests**

Add tests for:
- inherited self-attn/MLP/norm copied from scene decoder
- new cross-attn initialized separately
- cross-attn residual scalars zero-initialized
- stage-1 training policy freezing inherited blocks and enabling only new modules

**Step 2: Run test to verify it fails**

Run: `pytest debug/test_ho_decoder_initialization_policy.py -v`

Expected: FAIL because new initialization/training policy is not implemented

**Step 3: Write minimal implementation**

Implement:
- HO decoder init from scene decoder weights
- cross-attn init path
- staged freezing policy for new architecture

**Step 4: Run test to verify it passes**

Run: `pytest debug/test_ho_decoder_initialization_policy.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_ho_decoder_initialization_policy.py pi3/models/pi3x.py trainers/pi3x_training_policy.py
git commit -m "feat: add ho decoder init and staged training policy"
```

### Task 9: Run End-To-End Pi3X Smoke Verification

**Files:**
- Modify: `debug/test_pi3x_trainer_smoke.py`
- Modify: `debug/test_pi3x_config_smoke.py`

**Step 1: Write the failing tests**

Extend smoke coverage for:
- dense local hand/object token path
- retained scene branch
- retained object-multiview branch
- new prediction heads
- dense masked loss path

**Step 2: Run test to verify it fails**

Run: `pytest debug/test_pi3x_trainer_smoke.py debug/test_pi3x_config_smoke.py -v`

Expected: FAIL until all new interfaces are wired correctly

**Step 3: Write minimal implementation**

Update smoke fixtures and assertions to the new architecture.

**Step 4: Run test to verify it passes**

Run: `pytest debug/test_pi3x_trainer_smoke.py debug/test_pi3x_config_smoke.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add debug/test_pi3x_trainer_smoke.py debug/test_pi3x_config_smoke.py
git commit -m "test: update pi3x smoke coverage for local token ho redesign"
```
