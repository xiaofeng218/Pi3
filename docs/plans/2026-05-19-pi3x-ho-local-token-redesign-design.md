# Pi3X HO Local Token Decoder Redesign

**Goal:** Replace the current single-token hand/object HO pathway with a dense local-token pathway built from cropped hand/object regions, while keeping the scene decoder and object-multiview decoder intact.

**Non-Goal:** Do not preserve backward compatibility with the old single-token HO branch or sparse hand prediction pathway.

## 1. Scope

This redesign replaces:
- hand single-token query generation
- object single-token query generation
- sparse hand ownership/output flow

This redesign keeps:
- scene image/depth/intrinsics/poses inputs
- scene encoder/decoder
- object_multiview input and decoder
- multimodal depth/ray/pose conditioning on the scene branch

## 2. Input Contract

Keep model inputs unchanged:
- `imgs`
- `depths`
- `intrinsics`
- `poses`
- `hand_masks`
- `hand_is_right`
- `object_masks`
- `object_valid`
- `object_multiview`

Do not introduce a new `local_region_inputs` field.

`object_multiview` remains independent from object local crops:
- object local crop tokens replace the old object query token role
- object_multiview remains a separate modality and memory source

## 3. Scene Branch

Keep the current scene `encode()` and scene `decoder` unchanged in role:
- full-image DINOv2 encoding
- depth/ray additive conditioning
- pose injection blocks
- scene token decoding
- point/camera/conf/metric heads

The scene branch remains the global geometry backbone.

## 4. Local Crop Encoding

### 4.1 General Rule

Reuse the current `HandTokenAdapter` and `ObjectQueryAdapter` module positions, but fully replace their internal logic.

Both modules switch from:
- mask pooling on scene patch tokens

to:
- mask -> bbox
- bbox expansion
- crop from RGB image
- resize to fixed resolution
- encode crop with the external scene DINOv2 backbone
- output token sequences

### 4.2 Hand Crop

Use one shared crop box per hand region.

Apply the same crop box to two branches:
- HaMeR branch resize: `(256, 192)`
- DINO branch resize: `(224, 168)`

This keeps token counts aligned because:
- HaMeR patch size = `16`
- DINO patch size = `14`

### 4.3 Object Crop

Use object mask to crop the local object image.
Resize to `(224, 168)`.
Encode with the external scene DINOv2 backbone.
Output object local token sequence.

## 5. Hand Branch

Keep HaMeR.

Modify `HaMeRBackbone` / `HaMeREncoder` output semantics:
- old: `query + betas`
- new: `context tokens + betas`

`HandTokenAdapter` outputs DINO-based hand local tokens.

The two hand token streams must have identical token count:
- HaMeR tokens
- DINO hand crop tokens

Fuse hand tokens by:
- concat on channel dimension
- two-layer MLP
- output final hand token sequence

Keep `betas` from HaMeR.
`betas` are not predicted by the new hand heads.

## 6. Handedness

Use a three-level strategy:
- crop canonicalization
- explicit handedness embedding
- final left/right MANO selection

Canonicalization:
- right hand: keep as is
- left hand: horizontal flip

Apply canonicalization before both:
- HaMeR crop encoding
- DINO crop encoding

Keep `hand_is_right` as explicit `(B, N)` side condition.
Inject side embedding into both hand token streams before fusion.

At MANO expansion time, still choose:
- left MANO layer
- right MANO layer

according to `hand_is_right`.

## 7. Dense Token Organization

Replace sparse hand organization with dense `(B, N)` organization.

### 7.1 Hand

`hand_tokens`: `(B, N, T_hand, C)`

`hand_valid_mask`: `(B, N)`

### 7.2 Object

`object_tokens`: `(B, N, T_obj, C)`

`object_valid_mask`: `(B, N)`

No sparse hand entity structure remains inside the model.

### 7.3 Empty Tokens

Keep empty token sequences:
- `empty_hand_tokens`
- `empty_object_tokens`

If a view has no valid hand/object, fill that view with the corresponding empty token sequence.

Invalid views remain in the graph but are blocked from update by mask or `torch.where`.

## 8. HO Decoder Redesign

The new HO decoder keeps the layer schedule of the current scene decoder:
- 36 layers
- even layers: per-view layout
- odd layers: cross-view layout

The following branches all remain active and are updated layer-by-layer:
- scene hidden
- object_multiview hidden
- hand tokens
- object tokens

### 8.1 Even Layers

For each view:
1. hand tokens do self-attention only within that view
2. object tokens do self-attention only within that view
3. hand tokens cross-attend to that layer's scene tokens for the same view
4. object tokens cross-attend to that layer's scene tokens for the same view
5. add cross-attention result through learnable zero-initialized residual scaling
6. update scene hidden with the corresponding scene decoder block
7. update object_multiview hidden with its decoder block

Use separate learnable zero-initialized scalars for:
- hand-to-scene cross-attn residual
- object-to-scene cross-attn residual

### 8.2 Odd Layers

Across all views:
1. hand tokens do joint self-attention across views
2. object tokens do joint self-attention across views
3. object tokens cross-attend to object_multiview tokens
4. add cross-attention result through learnable zero-initialized residual scaling
5. update scene hidden
6. update object_multiview hidden

Hand tokens do not use extra memory cross-attention in odd layers.

## 9. HO Outputs and Prediction Heads

The HO branch output is directly:
- hand tokens
- object tokens

Then apply camera-head-style token aggregation logic:
- pool `hand_tokens` -> `pooled_hand_feature`
- pool `object_tokens` -> `pooled_object_feature`

Build three heads:
- `object_pose_head`
- `hand_global_head`
- `hand_pose_head`

`pooled_hand_feature` is shared by:
- `hand_global_head`
- `hand_pose_head`

`hand_pose_head` should preserve the old output semantics and produce the same rotmat-compatible MANO pose contract expected by the current downstream code.

`HandMANOHead` should no longer predict hand parameters directly. It should act only as a MANO wrapper that consumes:
- `global_orient`
- `hand_pose`
- `betas`
- `transl`
- `scale`
- `hand_is_right`

and then produces:
- `pred_mano_params`
- `pred_hand_mano_betas`
- MANO joints and vertices

Outputs should remain semantically close to the current training interface:
- `pred_object_rot6d`
- `pred_object_trans`
- `pred_object_scale`
- `pred_hand_transl`
- `pred_hand_scale`
- `pred_hand_mano_params`
- hand global rotation related outputs

## 10. Loss Migration

Move to dense `(B, N)` prediction and dense `(B, N)` GT.

Do not keep sparse hand supervision flow inside the model.

Use:
- `hand_valid_mask`
- `object_valid_mask`

to select valid views for supervision.

Trainer/GT pipeline should migrate from:
- dense GT -> sparse GT -> sparse pred alignment

to:
- dense GT -> dense pred -> valid-mask supervision

## 11. Initialization

For the new HO decoder:
- initialize self-attn, MLP, and norm from the corresponding pretrained scene decoder layer
- initialize new cross-attn modules normally
- initialize cross-attn residual scaling parameters to zero

If hand/object use separate block instances, both copy from the same corresponding scene decoder layer.

## 12. Training Strategy

### Stage 1

Train only newly introduced modules:
- cross-attn modules
- cross-attn residual scalars
- hand/object pooling heads
- `object_pose_head`
- `hand_global_head`
- `hand_pose_head`
- hand token fusion MLP
- side embeddings
- empty tokens

Freeze inherited self-attn, MLP, and norm copied from the scene decoder.

### Stage 2

Optionally unfreeze a subset of inherited modules if needed:
- norms
- MLPs
- or later decoder layers

Do not begin with full unfreeze.

## 13. Deliberate Removal of Old Logic

Remove or rewrite old logic directly:
- old hand single-token adapter behavior
- old object single-token query behavior
- sparse hand ownership-based internal prediction path
- compatibility paths intended only for the previous architecture

The redesign intentionally does not preserve old-path compatibility.

## 14. Validation

Need dedicated tests for:
- crop box generation
- hand left/right canonicalization
- HaMeR/DINO token count alignment
- dense `(B, N)` token organization
- invalid-view empty token stability
- HO decoder shape propagation
- output head shape and key contract
- dense loss masking correctness
