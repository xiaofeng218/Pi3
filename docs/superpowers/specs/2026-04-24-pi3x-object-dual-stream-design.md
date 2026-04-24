# Pi3X Object Dual-Stream Design

Date: 2026-04-24
Status: Draft for review
Scope: Shared-weight dual-stream Pi3X extension for object canonical multiview modeling

## 1. Purpose

This document defines how to extend `Pi3X` so that object canonical multiview images are modeled by the same backbone-decoder stack as the scene views, while keeping object- and hand-level cross-view interaction focused on instance tokens rather than full scene tokens.

The design adds:

- a shared-weight object stream fed by `DexYCB` canonical object multiview data
- per-view scene-side object query tokens
- asymmetric local attention for scene image blocks
- grouped global attention for scene, hand, and object branches
- an object pose head that regresses `obj2cam` rotation, translation, and scale from scene-side object queries

The design does not add:

- a second independent Pi3X parameter set
- full-sequence scene-object mixed global attention
- per-layer independent cross-attention parameter stacks
- object-flow point, camera, confidence, or metric supervision in the first version

## 2. Goals

- Reuse the existing Pi3X encoder, multimodal branches, decoder, and pose-inject machinery for object canonical views.
- Keep scene patch modeling separate from hand/object instance modeling during global interaction.
- Allow scene-side object queries to read canonical object multiview evidence every global stage.
- Preserve the current scene heads for points, camera, metric, and confidence with minimal semantic change.
- Keep tensor layout explicit enough that invalid hands or missing objects can be routed cleanly.

## 3. Non-Goals

- Designing a generic multi-instance object system beyond one object query per scene view.
- Adding object mesh decoding or dense object reconstruction.
- Letting hand/object tokens write back into scene patch tokens during local image attention.
- Introducing explicit canonical-view identity embeddings in the first version.
- Replacing the current hand path design.

## 4. DexYCB Object Data Contract Used by This Design

The new object path consumes fields already emitted by `DexYCBDataset`:

- current-scene object fields:
  - `object_multiview["grasped_object_id"]`
  - `object_multiview["grasped_object_mask"]`
  - `object_multiview["grasped_object_valid"]`
  - `object_multiview["grasped_object_pose_obj2cam"]`
- canonical object multiview fields:
  - `object_multiview["img"]`
  - `object_multiview["depthmap"]`
  - `object_multiview["camera_intrinsics"]`
  - `object_multiview["camera_pose"]`
  - `object_multiview["pts3d"]`

For the first implementation, only the mask, validity flag, canonical RGB, canonical depth, canonical intrinsics, and canonical poses are required by the model path. Ground-truth `grasped_object_pose_obj2cam` is used for the pose loss.

## 5. High-Level Architecture

The final architecture has two synchronized streams:

```text
Scene inputs (B, Ns, ...)
-> shared Pi3X encoder + multimodal branches
-> scene tokens: register + scene patches + hand queries + object query

Object canonical inputs (B, No=8, ...)
-> same shared Pi3X encoder + multimodal branches
-> object tokens: register + canonical object patches

decoder loop:
  image block:
    scene local asymmetric attention
    object local self-attention
  global block:
    scene global attention
    hand global attention
    object global interaction

heads:
  scene heads from scene patches
  hand head from hand path
  object pose head from scene-side object query
```

The key architectural decision is that the object stream is not a lightweight memory encoder. It is a full second stream through the shared Pi3X backbone-decoder path. The key interaction decision is that scene-side object queries can read canonical object evidence, but scene patches and canonical object patches are not globally mixed into one attention pool.

## 6. Shared Modules and New Modules

### 6.1 Shared modules

The following modules remain single-instanced and are used by both streams:

- RGB encoder
- depth encoder
- ray embedding
- decoder blocks
- pose inject blocks
- positional encoding utilities

This means the object stream does not duplicate Pi3X weights. The same module instance is called separately on scene and object tensors.

### 6.2 New modules

The design introduces the following new modules:

- `ObjectQueryAdapter`
  - builds one object query token per scene view from `grasped_object_mask`
- grouped attention routing inside decoder
  - dispatches local/global computations across scene, hand, and object branches
- `ObjectPoseHead`
  - regresses `rot6d + trans + scale` from final scene-side object query tokens

No standalone per-layer cross-attention stack is introduced in the first version. Instead, object interaction is implemented by grouped global attention rules.

## 7. Token Layout

Let:

- `B`: batch size
- `Ns`: number of scene views
- `No = 8`: number of canonical object views
- `P = H/14 * W/14 = 256`: patch tokens per view at `224x224`
- `R = 5`: register tokens
- `Kh <= 2`: hand query slots per scene view
- `Ko = 1`: object query slots per scene view
- `C = 1024`: token dimension

### 7.1 Scene stream

Per scene view:

```text
[register | scene patches | hand queries | object query]
```

Total length:

```text
Ts = R + P + Kh + Ko
```

### 7.2 Object stream

Per canonical object view:

```text
[register | canonical object patches]
```

Total length:

```text
To = R + P
```

The first version does not add an object-query token inside the canonical object stream.

## 8. Object Query Adapter

`ObjectQueryAdapter` builds scene-side object queries from scene RGB patch tokens and object masks.

Input:

```python
rgb_patch_tokens: Tensor[B, Ns, P, C]
grasped_object_mask: Tensor[B, Ns, H, W]
grasped_object_valid: Tensor[B, Ns]
```

Output:

```python
object_query: Tensor[B, Ns, 1, C]
object_query_pos: Tensor[B, Ns, 1, 2]
object_valid: Tensor[B, Ns]
```

Rules:

- if `grasped_object_valid=True` and the mask is non-empty:
  - downsample the mask to patch resolution
  - weighted-pool the scene RGB patch tokens
  - compute query position from the mask center in patch coordinates
- otherwise:
  - use a learnable `empty_object_query`
  - use a default position token, such as `(0, 0)`

The object query is a scene-side instance token. It is not allowed to inject its content back into scene patches during local image attention.

## 9. Decoder Scheduling

The current Pi3X decoder alternates between per-view image-stage attention and cross-view global-stage attention. The new design preserves that alternation but changes the token interaction pattern.

### 9.1 Image block

The image block runs separately on the scene stream and object stream.

#### Scene image block

For each scene view independently, tokens are partitioned into:

- `S`: scene tokens = `register + scene patches`
- `H`: hand queries
- `O`: object query

Attention is asymmetric:

- `S -> S`: allowed
- `S -> H/O`: blocked
- `H -> S`: allowed
- `O -> S`: allowed
- `H <-> H/O`: allowed in the same view
- `O <-> H/O`: allowed in the same view

Equivalent block mask:

```text
        keys
        S    H    O
q S   [ 1    0    0 ]
  H   [ 1    1    1 ]
  O   [ 1    1    1 ]
```

This means scene tokens remain a pure scene stream, while hand/object queries act as readers over same-view scene evidence.

#### Object image block

Each canonical object view independently runs ordinary self-attention over:

```text
[register | canonical object patches]
```

### 9.2 Global block

The global block is split into three logical computations rather than one pooled attention pass.

#### Scene global

Only scene tokens participate:

- `register + scene patches`
- hand queries excluded
- object query excluded

This branch handles scene-level cross-view geometric integration.

#### Hand global

Only hand queries participate:

- all valid hand queries across all scene views
- invalid hand slots are masked or gathered out

This branch handles hand instance cross-view aggregation.

#### Object global

Inputs:

- scene-side object queries
- canonical object multiview patch tokens

Interaction rule:

- scene-side object queries read canonical object tokens
- canonical object tokens do not query scene-side object queries

Canonical object tokens may still interact among themselves through the object stream's own local/global processing. The design requirement is only that they do not use scene-side object queries as keys or values for reverse interaction.

This branch handles object-instance alignment between scene observations and canonical object memory.

## 10. Pose Inject Usage

Both streams run pose-aware processing using the existing Pi3X pose inject path.

### 10.1 Scene stream pose inject

Uses the existing scene `poses` input.

The operation applies to the non-special scene token subset:

- scene patches
- hand queries
- object query

### 10.2 Object stream pose inject

Uses `object_multiview["camera_pose"]` as the canonical-view pose set.

The object stream therefore receives the same geometric treatment as the scene stream and does not require explicit canonical-view identity embeddings in the first version.

## 11. Identity and Validity Rules

### 11.1 Hand validity

- keep learnable empty hand tokens for slot stability
- keep explicit `hand_valid`
- in `hand_global`, invalid hand tokens are masked or gathered out

The empty token is used for layout stability and local-stage occupancy, not as a replacement for validity routing.

### 11.2 Hand handedness

- keep handedness embedding
- add it once when constructing hand query tokens
- do not re-add it at every decoder layer

### 11.3 Hand slot identity

- do not add hand-slot identity by default
- only add it later if slot semantics become stable and meaningful

### 11.4 Object validity

- keep learnable empty object query token
- keep explicit `object_valid`
- compute object loss only for valid object views

### 11.5 Canonical view identity

The first version does not add explicit canonical-view identity embeddings.

Rationale:

- the scene stream currently does not use explicit per-view identity embeddings
- object stream canonical views receive rays, poses, and pose inject
- this keeps the object stream semantically aligned with current Pi3X design

If canonical-view interaction later proves ambiguous or unstable, explicit view identity can be added as a targeted follow-up.

## 12. Output Head Design

Add a dedicated `ObjectPoseHead` that consumes the final scene-side object query token from each scene view.

Input:

```python
object_query_final: Tensor[B, Ns, C_or_2C]
```

Recommended output:

- `rot6d`: 6
- `trans`: 3
- `scale`: 1

Total:

```text
10 dims per scene view
```

The output is converted into:

- object rotation matrix
- object translation
- object scale
- assembled `obj2cam`

This head is independent from the scene point/camera/confidence heads, which continue to read only scene patch tokens.

## 13. Loss Design

### 13.1 Object pose loss

Use `grasped_object_pose_obj2cam` as supervision for valid object views only.

Recommended decomposition:

- rotation loss on the rotation decoded from `rot6d`
- translation loss on translation vector
- scale loss on predicted object scale

### 13.2 Validity routing

- if `object_valid=False`, object pose loss is zero for that view
- if `hand_valid=False`, hand-specific losses are zero for that slot

### 13.3 Scene losses

No changes to the first-version scene point, camera, confidence, or metric losses are required by this design.

## 14. Decoder Refactor Requirements

The current `Pi3X.decode()` must be refactored so that one loop iteration can operate on:

- scene-stream tensors
- object-stream tensors
- grouped local/global routing rules

The recommended internal boundary is:

- stream token construction
- image-stage scene block
- image-stage object block
- global-stage scene block
- global-stage hand block
- global-stage object block
- pose inject helper
- final stream-to-head slicing

This is a scheduling refactor, not a decoder-weight duplication.

## 15. Initial Implementation Scope

The first implementation should include:

- object query extraction from scene masks
- shared-weight object stream
- asymmetric scene local attention
- grouped scene/hand/object global routing
- object pose head and object-valid loss routing

The first implementation should exclude:

- explicit canonical-view identity embeddings
- object-stream auxiliary dense heads
- scene-patch updates from hand/object query content
- generalized multi-object support
- per-layer independent cross-attention parameter stacks

## 16. Risks and Mitigations

### 16.1 Risk: object global becomes too expensive

Mitigation:

- keep object global limited to scene-side object queries reading canonical object tokens
- avoid full mixed scene-object global attention

### 16.2 Risk: empty hand tokens contaminate hand global

Mitigation:

- keep `hand_valid`
- mask or gather invalid hand queries out of `hand_global`

### 16.3 Risk: local asymmetric attention requires custom masking

Mitigation:

- implement the scene image block as a masked-attention variant using the existing shared self-attention parameters
- avoid introducing a second local-attention module family

### 16.4 Risk: canonical view ambiguity without explicit view identity

Mitigation:

- rely first on rays, poses, and pose inject
- add explicit view identity only if experiments show ambiguity

## 17. Acceptance Criteria

The design is considered correctly implemented when:

- object canonical multiview images run through the same Pi3X backbone-decoder path as scene images
- scene local attention prevents scene tokens from reading hand/object query tokens
- hand and object queries are excluded from scene global attention
- valid hand queries aggregate only through `hand_global`
- scene-side object queries aggregate only through `object_global`
- object pose predictions are produced from scene-side object queries
- invalid hands and invalid objects are correctly masked from their losses

