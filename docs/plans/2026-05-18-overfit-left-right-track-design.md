# Overfit Left Right Track Design

## Goal

Make `train_overfit.sh` use exactly two explicitly configured DexYCB tracks: one left-hand track and one right-hand track.

## Current Behavior

The overfit config currently relies on `max_tracks: 1`, which truncates the DexYCB track list after split filtering. Because each DexYCB track carries a single `mano_side`, overfit training only sees one handedness.

## Proposed Design

Add an explicit track-selection config block to the DexYCB dataset configuration:

- `selected_tracks.left`
- `selected_tracks.right`

Each selector will contain:

- `subject`
- `sequence`
- `camera`

The dataset will build its normal split-filtered track list first, then resolve the configured selectors against that list and keep only the matched tracks. The resolver will validate that:

- each selector matches exactly one track
- the matched track handedness matches the selector key (`left` or `right`)

This keeps the behavior stable and readable, and avoids depending on implicit track ordering.

## Config Shape

The overfit config will declare real default tracks from the local DexYCB mirror:

- right: `20200709-subject-01 / 20200709_141754 / 836212060125`
- left: `20200709-subject-01 / 20200709_141931 / 836212060125`

`max_tracks: 1` will be removed. Batch size remains `1`, so training still overfits a tiny dataset but now alternates between two fixed samples.

## Testing

Add a dataset contract test that builds both left and right fixture tracks, configures explicit selectors, and verifies that exactly two tracks remain with handedness `["left", "right"]` or `["right", "left"]`.

Add a config smoke assertion that `overfit.yaml` now composes with `selected_tracks` instead of `max_tracks`.
