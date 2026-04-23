# HaMeREncoder Design

Date: 2026-04-23
Status: Draft for review
Scope: Isolated HaMeR-based sparse hand-query encoder for later Pi3X integration

## 1. Purpose

This document defines a new `HaMeREncoder` module under `pi3/models/hamer/`.

The module is intended to extract sparse hand query tokens from full-frame Pi3X inputs while keeping the implementation boundary strictly inside the HaMeR side of the codebase.

The module must:

- accept full-frame multi-view images
- accept sparse hand-instance masks as flat inputs
- crop and preprocess valid hand instances internally
- run HaMeR backbone and query decoding only
- return sparse hand queries with stable ownership metadata

The module must not:

- regress MANO pose, shape, or camera parameters
- invoke the MANO mesh layer
- repack outputs into dense `(B, N, M, ...)` tensors
- decide how hand tokens are fused into Pi3X

## 2. Goals

- Provide a clean reusable hand-query encoder with a narrow interface.
- Preserve sparse instance semantics end-to-end.
- Avoid computing backbone features for empty or invalid hand masks.
- Reuse as much of the existing local HaMeR implementation as possible.
- Keep later Pi3X-side dense refill and null-token handling outside this module.

## 3. Non-Goals

- Modifying `Pi3X` to consume hand tokens.
- Replacing the existing `HAMER` class inference path.
- Returning image features per hand instance.
- Returning MANO parameters or mesh outputs.
- Supporting arbitrary third-party runtime wrappers from `third_party/hamer/`.

## 4. High-Level Architecture

The new path is:

```text
imgs + hand_masks + owner_index + hand_is_right
-> valid-instance filtering
-> mask-to-bbox conversion
-> crop / resize / normalize / handedness canonicalization
-> HaMeRBackbone
-> sparse hand query tokens
```

The design introduces two new code units:

- `pi3/models/hamer/encoder.py`
  - contains `HaMeREncoder`
- `pi3/models/hamer/backbone_query.py`
  - contains `HaMeRBackbone`

No separate `HandInstanceExtractor` or `HandCropPreprocessor` class will be created. Those responsibilities stay as private helper functions inside `HaMeREncoder`.

## 5. Input Contract

`HaMeREncoder.forward(...)` will accept:

```python
imgs: Tensor[B, N, 3, H, W]
hand_masks: Tensor[K, H, W] or Tensor[K, 1, H, W]
owner_index: Tensor[K, 3]     # rows are (b, n, m)
hand_is_right: Tensor[K]      # bool-like
```

Definitions:

- `B`: batch size
- `N`: number of views per sample
- `K`: number of sparse hand instances provided to the encoder
- `m`: original per-image hand slot index, preserved for future refill on the Pi3X side

Requirements:

- `owner_index` must refer only to valid `(b, n)` image locations in `imgs`
- `hand_masks` and `owner_index` must be aligned row-wise
- `hand_is_right` must be aligned row-wise with the same ordering

The encoder will preserve the original sparse ordering and only remove invalid instances. Surviving outputs remain stable relative to the input order.

## 6. Output Contract

`HaMeREncoder.forward(...)` returns sparse results only:

```python
{
    "hand_queries": Tensor[K_valid, D],
    "owner_index": Tensor[K_valid, 3],
    "hand_is_right": Tensor[K_valid],
    "crop_boxes": Tensor[K_valid, 4],
}
```

Where:

- `K_valid <= K`
- `D` is the HaMeR query token dimension

This module will not output:

- dense refill tensors
- null / empty hand tokens
- validity masks for dense layouts

The Pi3X side can derive dense validity later from sparse `owner_index`.

## 7. Internal Module Design

### 7.1 HaMeREncoder

`HaMeREncoder` is the public sparse encoder wrapper.

Responsibilities:

- validate sparse inputs
- normalize mask shapes
- filter empty or too-small masks
- compute bounding boxes from masks
- gather source images from `imgs` using `owner_index`
- crop, rescale, resize, and normalize hand images
- canonicalize left/right hands
- call `HaMeRBackbone`
- return sparse outputs with stable ownership metadata

Recommended constructor inputs:

- `cfg`
- `rescale_factor`
- `min_mask_area`

Private helpers inside `HaMeREncoder` should include:

- `_normalize_mask_shape`
- `_compute_bbox_from_mask`
- `_filter_valid_instances`
- `_gather_source_images`
- `_crop_and_resize`
- `_normalize_input`
- `_canonicalize_handedness`

### 7.2 HaMeRBackbone

`HaMeRBackbone` is a simplified name for the internal HaMeR query-producing stack.

Responsibilities:

- own the local HaMeR ViT backbone
- own the single-query transformer decoder path
- output only the final query token embedding

It must not:

- regress `pose / shape / cam`
- instantiate MANO layers
- return mesh outputs

Conceptually:

```text
crop image
-> ViT feature map
-> flatten to context tokens
-> query token initialization
-> transformer decoder cross-attention
-> query token output
```

## 8. Reuse Strategy from Existing HaMeR Code

### 8.1 Reused components

The following existing modules should be reused directly:

- `pi3/models/hamer/backbones/vit.py`
- `pi3/models/hamer/components/pose_transformer.py`

### 8.2 Query-path extraction

The current `MANOTransformerDecoderHead` in `pi3/models/hamer/heads/mano_head.py` combines:

1. image feature to query token decoding
2. query token to MANO / camera regression

The new design separates these concerns.

`HaMeRBackbone` should keep only the first part:

- feature map rearrange
- mean-shape or zero query-token setup
- decoder call
- `token_out.squeeze(1)`

The following regression heads are intentionally excluded:

- `decpose`
- `decshape`
- `deccam`

No changes are required to the existing `HAMER` class for this design phase.

## 9. Cropping and Preprocessing Design

Cropping behavior should follow the method used in:

- [third_party/hamer/demo_mask_bbox_dexycb.py](/home/hanxiaofeng/Pi3-training/third_party/hamer/demo_mask_bbox_dexycb.py:50)

The reference method is used as a design guide, not as a runtime dependency.

### 9.1 Bounding box extraction

Bounding boxes are computed from binary hand masks using the min/max occupied coordinates:

```text
xyxy = [xmin, ymin, xmax, ymax]
```

Empty masks are invalid.

### 9.2 Validity filtering

An instance is valid only if:

- the mask is non-empty
- the mask area is at least `min_mask_area`

This avoids sending tiny noisy masks through the backbone.

### 9.3 Box expansion

Bounding boxes should be expanded by a configurable `rescale_factor`, matching the spirit of the HaMeR demo path.

The expanded box must be clamped to the image bounds before cropping.

### 9.4 Resize and normalization

Crops must be resized to the HaMeR expected image size. The current local HaMeR backbone uses an effective crop width compatible with the `256x192` path seen in the existing code.

Normalization should match the local HaMeR preprocessing convention rather than Pi3X image normalization.

## 10. Handedness Handling

`hand_is_right` is part of the sparse instance input and will also be preserved in the sparse output.

Inside `HaMeREncoder`, left-hand crops should be horizontally flipped so the backbone sees a canonical hand orientation.

This gives the query token a more stable semantic space while preserving the original handedness metadata externally.

Rules:

- if `hand_is_right == True`, keep crop orientation
- if `hand_is_right == False`, horizontally flip crop before encoding

The output `hand_is_right` remains the original flag, not the canonicalized flag.

## 11. Ordering Guarantees

The encoder must preserve the original input order.

Behavior:

- instances are inspected in the incoming row order `0..K-1`
- invalid rows are dropped
- valid rows keep their original relative order

This ordering guarantee is important because later Pi3X-side refill logic may assume deterministic sparse row ordering during debugging and test assertions.

## 12. Error Handling

The encoder should distinguish between:

- malformed inputs
- valid-but-empty runtime cases

Recommended behavior:

- raise on shape mismatch between `hand_masks`, `owner_index`, and `hand_is_right`
- raise on invalid owner indices that point outside `(B, N)`
- return empty sparse outputs when `K == 0`
- return empty sparse outputs when all instances are filtered as invalid

Returning empty sparse outputs is preferred over throwing for the “no valid hands” case.

## 13. File-Level Change Plan

### 13.1 New files

- `pi3/models/hamer/encoder.py`
- `pi3/models/hamer/backbone_query.py`

### 13.2 Modified files

- `pi3/models/hamer/__init__.py`
  - export `HaMeREncoder`

This design does not require modifying `pi3/models/hamer/heads/mano_head.py`. The new query-only path should be implemented independently so the existing MANO regression path remains unchanged.

## 14. Testing Strategy

Testing is required once implementation begins.

### 14.1 Unit-level smoke test with random data

A dedicated smoke test should construct random inputs that satisfy the interface:

- random `imgs` with shape `(B, N, 3, H, W)`
- random sparse masks with a mix of:
  - valid masks
  - empty masks
  - tiny masks below threshold
- aligned `owner_index`
- aligned `hand_is_right`

Assertions:

- the module runs without shape errors
- invalid instances are dropped
- output ordering is preserved
- output `owner_index` and `hand_is_right` remain aligned with surviving rows
- `hand_queries.shape[0] == len(output["owner_index"])`

This smoke test is specifically required by the user and is part of completion criteria for implementation.

### 14.2 Deterministic ordering test

Construct a small hand-crafted sparse batch where only selected rows are valid.

Assert that:

- the valid rows survive in their original relative order
- no sorting or regrouping happens internally

### 14.3 Empty-batch test

Run with:

- `K == 0`, and separately
- `K > 0` but all masks invalid

Assert empty sparse outputs without crashing.

### 14.4 Minimal integration-prep test

Without integrating into Pi3X yet, verify the sparse output is sufficient for later dense refill:

- `owner_index` is preserved
- no hidden dense state is required from `HaMeREncoder`

## 15. Open Design Constraints

The following decisions are intentionally fixed for this design:

- sparse in, sparse out
- no dense refill in HaMeR
- no null-hand token in HaMeR
- no explicit dense validity mask output
- stable input-order preservation
- crop logic stays inside `HaMeREncoder`

These constraints reduce scope and keep the module reusable.

## 16. Recommendation

Implement `HaMeREncoder` as a sparse, order-preserving, HaMeR-side query encoder that owns crop preprocessing and exposes only sparse hand query results. Keep all dense refill and Pi3X fusion logic out of this module.
