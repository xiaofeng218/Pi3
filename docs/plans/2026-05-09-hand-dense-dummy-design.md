# Dense Dummy Hand Slot Design

**Problem**

Dual-GPU training can hang when one rank has no valid hand tokens for a step while another rank still executes the hand branch. The current hand branch has an `empty_hand_token`, but that placeholder only reaches the decoder. It does not flow through `hand_mano_head` or hand losses, so hand-related parameters can be unused on some ranks and used on others in the same step.

**Goal**

Keep the hand branch semantically inactive when a batch has no real hands, while ensuring the hand branch still participates in the computation graph on every rank and every step.

**Chosen Approach**

Use dynamic per-batch hand slot count as today, but force a single dummy hand slot when a batch has no hands:

- If a batch contains real hands, keep `num_hand_tokens = max_slot + 1`.
- If a batch contains no hands, return `num_hand_tokens = 1`.
- Always run the dense hand slots through `decode()` and `hand_mano_head`.
- Always emit dense `pred_hand_*` outputs.
- Use a dense `hand_valid_mask` so invalid/dummy slots contribute zero hand loss.

**Why This Approach**

- Preserves the current dynamic-slot behavior for the common single-hand case.
- Avoids the long-term compute overhead of always padding to `max_num_hands`.
- Produces a stable DDP/autograd graph because the hand head always executes.
- Avoids the brittleness of zero-anchor-only solutions.

**Architecture**

1. `HandTokenAdapter`
   - Keep dynamic dense slot packing for real hands.
   - When there are no real hands, return one dense dummy slot instead of `None`.
   - Return a dense validity mask with all `False`.

2. `Pi3XTrainer.forward_batch()`
   - Stop collapsing empty hand encoder outputs to `None`.
   - Instead, feed the model a dense dummy hand path when the batch has no real hands.
   - Build dense hand GT tensors aligned to `(B, N, M, ...)`, plus dense validity masks.

3. `Pi3X.forward()`
   - Continue decoding dense hand tokens.
   - Replace the current sparse-only hand head path with a dense head path.
   - Flatten `(B, N, M, C)` to `(B*N*M, C)` for `hand_mano_head`, then reshape outputs back to dense form.
   - Keep optional sparse debug outputs only if they are still needed elsewhere.

4. `HandObjectLoss`
   - Consume dense hand predictions and dense hand GT.
   - Hand losses always exist, but invalid slots are masked to zero.
   - Fully dummy batches return zero-valued hand losses that still remain connected to the graph.

**Data Shape Contract**

- Dense hand token features: `(B, N, M, C)`
- Dense hand valid mask: `(B, N, M)`
- Dense hand predictions:
  - `pred_hand_transl`: `(B, N, M, 3)`
  - `pred_hand_scale`: `(B, N, M, 1)` or log-scale equivalent
  - `pred_hand_mano_betas`: `(B, N, M, 10)`
  - `pred_hand_mano_params.global_orient`: `(B, N, M, 1, 3, 3)`
  - `pred_hand_mano_params.hand_pose`: `(B, N, M, J, 3, 3)`
  - `pred_hand_joints_3d`: `(B, N, M, 21, 3)`
  - `pred_hand_vertices`: `(B, N, M, 778, 3)`

**Compatibility Notes**

- Existing code paths that assume sparse hand predictions will need to be updated or given an explicit sparse conversion helper.
- Logging and visualization can still derive sparse valid hands from the dense mask if needed.
- `find_unused_parameters=true` should remain enabled until the dense path is implemented and verified under dual-GPU.

**Validation Criteria**

- Dual-GPU training no longer hangs at steps where one rank has no real hands and another does.
- Hand losses are numerically zero for fully dummy batches.
- Hand branch outputs always exist, even on no-hand batches.
- Single-hand batches still use dynamic slot count rather than fixed full padding.
