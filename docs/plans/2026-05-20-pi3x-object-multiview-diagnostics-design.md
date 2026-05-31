# Pi3X Object Multiview Diagnostics Design

**Goal:** Add an offline-only diagnostic path that feeds DexYCB `object_multiview` inputs into a pretrained Pi3X image backbone, aligns predicted point clouds with predicted camera poses, and exports a rerun scene to judge whether object-only inputs are handled coherently.

**Scope:** This is a debug workflow only. It must not change the training path, checkpoint format, or the existing Pi3X hand-object trainer behavior.

## Context

The codebase already contains the required pieces in separate places:

- `datasets.dexycb_dataset.DexYCBDataset` can expose shared `object_multiview` payloads containing `img`, `depthmap`, `camera_intrinsics`, and `camera_pose`.
- `pi3.models.pi3.Pi3.forward()` already predicts `local_points`, `camera_poses`, and world-space `points` from image-only multiview inputs.
- Existing rerun export helpers already visualize DexYCB point clouds and Pi3X predictions, but there is no focused object-only diagnostic pipeline tying those paths together.

The question to answer is narrow: when Pi3X is driven only by DexYCB object multiview RGB inputs, do the predicted local geometry and camera poses produce a coherent fused point cloud?

## Requirements

- Reuse a pretrained Pi3X checkpoint for inference only.
- Use DexYCB `object_multiview` inputs as the model input source.
- Restrict the pipeline to image-only Pi3 inference. Do not invoke the HO branch.
- Export enough geometry to compare:
  - GT depth point clouds aligned with GT camera poses.
  - Predicted local points aligned with predicted camera poses.
  - Optional cross-checks using predicted local points aligned with GT poses.
- Emit a rerun `.rrd` plus lightweight JSON metadata/statistics.
- Keep the implementation isolated to debug tooling and tests.

## Non-Goals

- No trainer refactor.
- No new production inference mode.
- No checkpoint conversion.
- No attempt to improve model quality.
- No new dataset format.

## Approach Options

### Option 1: Standalone debug script

Build a dedicated script that loads DexYCB samples, extracts `object_multiview`, forwards them through the pretrained Pi3 model, computes aligned point clouds, and exports rerun artifacts.

Pros:

- Minimal risk to training code.
- Easy to iterate on diagnostics.
- Keeps object-only assumptions local.

Cons:

- Some setup code must be duplicated or lightly wrapped from trainer utilities.

### Option 2: Reuse Pi3X trainer end-to-end

Create a trainer-backed diagnostic mode that loads a batch and then routes only the object multiview payload through the Pi3 image model.

Pros:

- Reuses existing config and checkpoint wiring.

Cons:

- Pulls in trainer concerns that are irrelevant to the diagnostic.
- Higher risk of accidental coupling to HO assumptions.

### Option 3: Add a formal object-only inference mode

Extend Pi3X or the trainer with a new public mode.

Pros:

- Potentially reusable later.

Cons:

- Overbuilt for the current question.
- Changes production behavior surface.

## Recommended Design

Choose Option 1: a standalone debug script with small helper functions and focused tests.

The script will:

1. Load one DexYCB batch/sample with `include_object_multiview_payload=true`.
2. Extract the shared object multiview RGB, depth, intrinsics, and GT camera poses.
3. Load the pretrained Pi3/Pi3X image model checkpoint.
4. Run image-only multiview forward inference on the object multiview RGB tensor.
5. Build three geometry sets:
   - `gt_world_points`: depth backprojection aligned by GT `camera_pose`.
   - `pred_world_points`: predicted `local_points` aligned by predicted `camera_poses`.
   - `pred_local_points_with_gt_pose`: predicted `local_points` aligned by GT `camera_pose` for pose-vs-geometry debugging.
6. Export a rerun scene containing per-view images plus the three fused point-cloud variants.
7. Emit `meta.json` and `stats.json` summarizing sample identity, tensor shapes, valid point counts, and basic pose/scale ranges.

## Data Flow

### Input

- Dataset source: DexYCB subject + split + batch/sample selection.
- Modalities read from `object_multiview`:
  - `img`
  - `depthmap`
  - `camera_intrinsics`
  - `camera_pose`

### Model Input

- Only the `img` tensor is fed into `Pi3.forward()`.
- Depth and GT camera poses are used for reference geometry and comparison only.

### Derived Outputs

- `pred.local_points`
- `pred.camera_poses`
- `pred.points`
- GT world-space point clouds from depth backprojection
- Fused per-mode point clouds for rerun

## Rerun Layout

The export should make failure modes visually obvious.

- `frames/<view>/rgb`
- `frames/<view>/depth`
- `world/gt_points`
- `world/pred_points`
- `world/pred_points_gt_pose`
- `world/cameras/gt/<view>`
- `world/cameras/pred/<view>`

If one layer is empty, the export should log a clear fallback rather than crash.

## Diagnostics To Compare

The visualization is expected to separate three failure classes:

1. GT fused point clouds are coherent, but predicted fused point clouds scatter:
   the main problem is camera pose prediction.
2. GT fused point clouds are coherent, and `pred_local_points_with_gt_pose` looks reasonable, but `pred_world_points` scatter:
   local geometry is usable, but pose prediction is unstable.
3. Both GT-aligned and pred-aligned predicted clouds look poor:
   object-only RGB inputs do not match the pretrained Pi3 geometry assumptions well enough.

## Implementation Notes

- Prefer a new debug script instead of modifying trainer code.
- Reuse existing environment setup and DexYCB root conventions from current debug utilities.
- Keep tensor shaping explicit. Pi3 expects `B x N x 3 x H x W`.
- If the checkpoint is wrapped by a larger Pi3X module, isolate and load the Pi3 image submodule rather than invoking HO-specific heads.
- Favor small local helper functions over broad abstractions.

## Testing Strategy

Add narrow unit tests for:

- Extracting object multiview tensors into the expected Pi3 input shape.
- Building GT world points from depth + intrinsics + pose.
- Selecting and labeling rerun geometry payloads.
- Validating that the diagnostic chooses the Pi3 image model path rather than the HO branch.

Do not depend on full DexYCB assets in unit tests. Use synthetic tensors.

## Risks

- The pretrained checkpoint may be stored under Pi3X naming rather than bare Pi3 naming.
- DexYCB object multiview image resolution may differ from the checkpoint's expected resolution.
- The camera pose convention in DexYCB must match the convention used by Pi3 world-space outputs.
- Point count may be too large for comfortable rerun browsing unless downsampled.

## Mitigations

- Add a checkpoint-loading helper with explicit error messages about missing image-model weights.
- Surface tensor shapes and pose ranges into `stats.json`.
- Start with one sample and one export path before adding broader iteration.
- Add optional random/fixed downsampling for rerun point clouds if needed.
