# Pi3X Hand/Object Training Framework Design

Date: 2026-04-24
Status: Draft for review
Scope: Single-stage Pi3X training framework for joint hand/object optimization

## 1. Purpose

This document defines the training framework for the current Pi3X hand/object setup.

The goal is to add a dedicated training path that can:

- train hand and object outputs together in every batch
- freeze the Pi3X scene backbone and keep the optimization surface narrow
- apply LoRA-based fine-tuning to `ho_decoder`
- connect the existing hand/object loss helper to the trainer
- provide a clean config, logging, checkpointing, and validation flow

This design is intentionally minimal. It does not add:

- stage-wise curriculum training
- separate hand-only or object-only training branches
- multi-task routing across different objectives
- scene-head fine-tuning beyond the frozen backbone policy

## 2. Goals

- Keep the training path explicit and easy to reason about.
- Use one batch, one model, one loss, and one optimizer step.
- Make trainable parameters small and intentional.
- Preserve existing generic trainer infrastructure where possible.
- Keep logging and validation sufficient for debugging 3D hand/object optimization.

## 3. Non-Goals

- Reworking the dataset format.
- Rewriting the generic `BaseTrainer` stack.
- Adding a second training phase or warmup schedule.
- Designing new losses beyond the already agreed hand/object loss helper.
- Adding 2D reprojection supervision for this first version.

## 4. Current State Summary

The repository already contains a reusable training skeleton:

- `scripts/train_pi3.py` is the Hydra entry point.
- `trainers/base_trainer_accelerate.py` provides accelerator setup, optimizer/scheduler setup, epoch loops, and checkpointing.
- `trainers/pi3_trainer.py` is an older Pi3-specific trainer.

However, the current trainer path is still tied to the old Pi3 configuration and does not yet represent the current Pi3X hand/object training target.

The new framework therefore introduces a dedicated Pi3X training specialization instead of trying to overload the old trainer.

## 5. Training Contract

### 5.1 Batch contract

Training is joint by construction.

Every training batch must contain both:

- hand supervision
- object supervision

The trainer does not route samples into separate branches. If a sample lacks either component, it should be filtered at dataset or collation time rather than split into a special-case training path.

### 5.2 Model contract

The model is the current Pi3X hand/object architecture with:

- frozen scene encoder
- frozen scene decoder
- frozen multimodal depth/ray embeddings by default
- trainable hand/object heads
- trainable adapters and learnable tokens
- trainable `ho_decoder` LoRA parameters plus a small set of explicitly listed norm / scale / bias parameters

### 5.3 Loss contract

The trainer uses a single loss module:

- `HandObjectLoss`

The loss module is responsible for:

- estimating scene depth scale alignment
- computing hand losses
- computing object losses
- returning a scalar total loss plus a dict of loggable metrics

The trainer should not re-implement geometric logic.

## 6. Architecture

The training graph is:

```text
DexYCBDataset batch
  -> Pi3X.forward(...)
  -> HandObjectLoss(...)
  -> optimizer.step()
```

The training framework is split into five focused units:

1. `Pi3XTrainer`
   - owns the loop, optimizer stepping, validation, checkpointing, and logging
2. freeze / unfreeze policy
   - determines which modules are trainable
3. optimizer group builder
   - assigns learning rates and weight decay rules
4. `HandObjectLoss`
   - computes the actual supervision terms
5. visualization / debug export hooks
   - produce train-time outputs for inspection

The design keeps each unit small enough that it can be verified independently.

## 7. Trainable Parameter Policy

### 7.1 Frozen modules

The following are frozen by default:

- `encoder`
- `decoder`
- `depth_encoder`
- `depth_emb`
- `ray_embed`
- `point_decoder`
- `point_head`
- `camera_decoder`
- `camera_head`
- `metric_decoder`
- `metric_head`
- `conf_decoder`
- `conf_head`
- `MANO layer` parameters
- any legacy scene-only head not used by hand/object optimization

### 7.2 Trainable modules

The following are trainable by default:

- `hand_token_adapter`
- `object_query_adapter`
- `hand_mano_head`
- `object_pose_head`
- empty / learnable hand-object query tokens owned by the adapters or heads
- `register_token`
- `metric_token`
- `ho_decoder` LoRA parameters
- selected non-LoRA `ho_decoder` scalars only if explicitly enumerated in the freeze policy

The `depth_emb` and `ray_embed` modules are explicitly frozen in the first version.

### 7.3 `ho_decoder` fine-tuning policy

The recommended design is:

- keep the base `HOBlockRope` weights frozen
- add LoRA only to the large projection paths
- train only the LoRA parameters plus a small set of non-LoRA scalar parameters

The first-pass LoRA placement should prioritize:

- `cross_attn` projections inside `HOBlockRope`
- optionally `self_attn` projections if needed later
- optionally `mlp` projections if cross-attn-only adaptation is insufficient

The first-pass non-LoRA trainable `ho_decoder` parameters should be limited to:

- `LayerNorm` affine weights and biases
- `LayerScale` scalars
- projection and MLP biases if they are explicitly left trainable

This design is preferred over “add LoRA to every parameter and only train LoRA” because:

- normalization and scale parameters do not benefit from low-rank decomposition
- cross-attention is the primary adaptation path for hand/object interaction
- the framework stays easier to debug with fewer moving parts

## 8. Optimizer Design

The optimizer is `AdamW`.

The recommended parameter groups are:

1. hand/object heads
2. adapters + learnable tokens
3. `ho_decoder` LoRA parameters

Optional additional separation:

- a lower learning rate for small bridge parameters such as tokens and scale-related scalars

The design should keep weight decay disabled for one-dimensional tensors and bias parameters, consistent with the existing optimizer helper behavior.

## 9. Forward / Loss Flow

### 9.1 Forward path

`Pi3X.forward(...)` should return the existing unified hand/object prediction dictionary.

The trainer should not unpack the model output into separate task-specific forward branches.

### 9.2 Loss path

`HandObjectLoss(pred, batch)` computes:

- scene scale factor from depth alignment
- hand losses
- object losses
- loggable sub-loss scalars

The trainer only sees:

- `loss`
- `details`

The trainer should treat the loss as a black box beyond moving tensors to device and handling validation mode.

## 10. Loss Surface

This training framework assumes the loss definitions already agreed in the previous design:

### 10.1 Hand

- `transl_dir`
- `transl_scale`
- `scale`
- root-relative `joints_3d`
- root-relative `vertices_3d`

### 10.2 Object

- `rot6d`
- `transl_dir`
- `transl_scale`
- `scale`

The framework does not add additional 2D supervision in this first version.

## 11. Configuration Layout

The design introduces a dedicated Pi3X training config set.

### 11.1 Model config

Suggested file:

- `configs/model/pi3x_hand_object.yaml`

Responsibilities:

- instantiate Pi3X
- configure hand/object modules
- configure MANO layer injection
- configure `ho_decoder` LoRA settings if exposed at model construction time

### 11.2 Loss config

Suggested file:

- `configs/loss/hand_object_loss.yaml`

Responsibilities:

- loss weights
- scene scale estimation settings
- toggles for optional loss components

### 11.3 Train config

Suggested file:

- `configs/train/train_pi3x_hand_object.yaml`

Responsibilities:

- batch size
- epochs
- optimizer
- scheduler
- gradient clipping
- checkpoint interval
- freeze policy toggles
- validation cadence

### 11.4 Entry config

The existing `scripts/train_pi3.py` entry point can remain unchanged.

Hydra should route to the new trainer and the new config set through config composition rather than a new script.

## 12. Trainer Design

### 12.1 `Pi3XTrainer`

A new trainer should subclass the existing base trainer.

Responsibilities:

- build the model
- apply the freeze policy before optimizer creation
- build optimizer param groups
- instantiate `HandObjectLoss`
- run train / validation epochs
- save checkpoints
- log scalar metrics and summaries

### 12.2 Required trainer hooks

The trainer should implement or override:

- `prepare_model()`
- `build_optimizer()`
- `before_epoch()`
- `forward_batch()`
- `calculate_loss()`
- validation / debug export hooks

### 12.3 Validation behavior

Validation should remain lightweight and deterministic.

The first version only needs to verify:

- forward pass runs
- loss is finite
- checkpoint save / restore works
- loggable metrics are emitted

## 13. Logging and Visualization

### 13.1 Scalar logging

The following scalars should be logged:

- total loss
- hand sub-losses
- object sub-losses
- scene scale factor
- hand/object transl scale terms
- hand/object scale terms
- learning rate
- gradient norm

### 13.2 Visual logging

The first version should keep visualization simple:

- input RGB
- hand/object masks
- depth alignment comparison
- predicted hand/object 3D summaries

The trainer should not be coupled to a heavyweight 3D viewer. Visualization hooks should be optional and easy to disable.

## 14. Checkpointing and Resume

Checkpointing should reuse the existing accelerator-based save / restore flow.

The framework must preserve:

- model weights
- optimizer state
- scheduler state
- current epoch / global step

The design should also make trainable parameter intent recoverable through the config, not only through the checkpoint.

## 15. Verification Plan

The first implementation should be verified with:

1. one forward-only smoke test
2. one loss smoke test
3. one optimizer-group smoke test
4. one checkpoint save / load smoke test
5. one tiny overfit test on a handful of samples

The minimum acceptable result is:

- no NaNs in loss
- no shape mismatches in the new trainer path
- gradients flowing to the intended trainable modules only

## 16. Risks

- Over-freezing may make `ho_decoder` adaptation too weak.
- Over-enabling LoRA may make debugging harder without meaningful accuracy gain.
- If dataset samples are not guaranteed to contain both hand and object labels, the single-target assumption will fail and the dataset contract must be fixed before training.
- Logging too much visual data too early may slow the training loop unnecessarily.

## 17. Implementation Boundary

This document covers the training framework only.

It does not modify:

- the Pi3X inference semantics
- the hand/object output interfaces
- the already agreed loss formulas
- the dataset content definitions

The implementation should stay inside the training boundary and use the existing model and loss helpers as inputs.
