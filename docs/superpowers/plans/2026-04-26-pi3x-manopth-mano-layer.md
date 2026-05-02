# Pi3X MANO Layer Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Pi3X hand geometry backend with a vendored `manopth`-compatible MANO layer while keeping `HandMANOHead` semantics aligned and removing all outer left/right flip handling.

**Architecture:** Move the necessary `manopth` MANO implementation into `pi3/models/hamer` as the canonical backend. Expose a side-aware layer builder/cache that `Pi3X`, `HandMANOHead`, `HandObjectLoss`, and visualization/debug code can share. Preserve the current head output contract (`pred_hand_mano_params`, `pred_hand_vertices`, `pred_hand_joints_3d`, translation/scale fields), but route handedness and translation entirely through the MANO backend.

**Tech Stack:** PyTorch, existing Pi3 model code, vendored `manopth` math helpers, pytest/unittest.

---

### Task 1: Vendor the MANO backend into `pi3/models/hamer`

**Files:**
- Create: `pi3/models/hamer/mano_layer.py`
- Modify: `pi3/models/hamer/__init__.py`
- Modify: `pi3/models/hamer/hand_mano_head.py`
- Modify: `pi3/models/hamer/mano_wrapper.py` or deprecate it if no longer used

- [ ] **Step 1: Write the failing test**

Create a focused unit test that imports the new vendored layer and verifies it can be instantiated with `side="right"` and `side="left"`, exposes `th_faces`, and returns `(verts, joints)` tensors when called with MANO pose/betas/transl tensors.

```python
def test_vendored_mano_layer_instantiates_and_forwards():
    right = ManoLayer(side="right", mano_root=mano_root, use_pca=True, flat_hand_mean=False, ncomps=45)
    left = ManoLayer(side="left", mano_root=mano_root, use_pca=True, flat_hand_mean=False, ncomps=45)

    verts_r, joints_r = right(pose, betas, transl)
    verts_l, joints_l = left(pose, betas, transl)

    assert verts_r.shape[-1] == 3
    assert joints_r.shape[-1] == 3
    assert hasattr(right, "th_faces")
    assert hasattr(left, "th_faces")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest debug/test_pi3x_hand_modules.py -k vendored_mano_layer -v`
Expected: fail because `pi3.models.hamer.mano_layer` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Copy the necessary `manopth` math helpers into `pi3/models/hamer` and implement a vendored `ManoLayer` that keeps the current inspect/debug constructor semantics:

```python
class ManoLayer(nn.Module):
    def __init__(self, center_idx=None, flat_hand_mean=True, ncomps=6, side="right", mano_root="mano/models", use_pca=True, root_rot_mode="axisang", joint_rot_mode="axisang", robust_rot=False):
        ...

    def forward(self, th_pose_coeffs, th_betas=torch.zeros(1), th_trans=torch.zeros(1), root_palm=torch.Tensor([0]), share_betas=torch.Tensor([0])):
        ...
        return th_verts, th_jtr
```

Make the new module self-contained so `pi3` no longer depends on `third_party/manopth` at runtime.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest debug/test_pi3x_hand_modules.py -k vendored_mano_layer -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pi3/models/hamer/mano_layer.py pi3/models/hamer/__init__.py pi3/models/hamer/mano_wrapper.py debug/test_pi3x_hand_modules.py
git commit -m "feat: vendor manopth mano layer into pi3"
```

### Task 2: Make `HandMANOHead` side-aware and remove outer flipping

**Files:**
- Modify: `pi3/models/hamer/hand_mano_head.py`
- Modify: `pi3/models/pi3x.py`
- Modify: `debug/test_pi3x_hand_modules.py`

- [ ] **Step 1: Write the failing test**

Extend the hand-head test to verify that `hand_is_right=False` causes the head to use the left MANO backend without any external geometry mirror helper.

```python
def test_hand_mano_head_uses_side_specific_layer_without_flip():
    head = HandMANOHead(cfg, in_dim=16, hidden_dim=8, mano_layer_factory=factory)
    out = head(hand_tokens, hand_is_right=torch.tensor([False]))
    assert "pred_hand_vertices" in out
    assert "pred_hand_joints_3d" in out
    assert factory.calls == ["left"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest debug/test_pi3x_hand_modules.py -k hand_mano_head_uses_side_specific_layer_without_flip -v`
Expected: fail because `HandMANOHead` still mirrors externally and does not accept a side-aware backend.

- [ ] **Step 3: Write minimal implementation**

Update `HandMANOHead` so it:
- keeps the current output keys and shape contract
- accepts a side-aware MANO backend or factory
- routes each sample to the correct side-specific layer
- passes `hand_transl` into the MANO backend
- removes `_mirror_handed_geometry` and any caller-side flip logic

Update `Pi3X` so it builds the new vendored layer and passes the side-aware backend into `HandMANOHead` without any outer handedness correction.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest debug/test_pi3x_hand_modules.py -k hand_mano_head_uses_side_specific_layer_without_flip -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pi3/models/hamer/hand_mano_head.py pi3/models/pi3x.py debug/test_pi3x_hand_modules.py
git commit -m "feat: route hand head through side-aware mano backend"
```

### Task 3: Make loss, trainer, and visualization consume the same side-aware MANO backend

**Files:**
- Modify: `pi3/models/hand_object_loss.py`
- Modify: `trainers/pi3x_trainer.py`
- Modify: `pi3/visualization/pi3x_rerun_export.py`
- Modify: `debug/inspect_pi3x_pred_vs_gt.py`

- [ ] **Step 1: Write the failing test**

Add a test that builds GT hand vertices from `hand_pose_mano`, `hand_mano_betas`, and `hand_is_right`, then checks that the loss path uses the matching left/right backend and still computes root-relative geometry losses.

```python
def test_hand_object_loss_uses_hand_side_for_gt_geometry():
    loss = HandObjectLoss(mano_layer_factory=factory)
    total, details = loss(pred, gt)
    assert factory.calls == ["right", "left"]
    assert "hand_vertices_loss" in details
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest debug/test_hand_object_loss.py -v`
Expected: fail because the loss currently does not receive side-aware MANO construction.

- [ ] **Step 3: Write minimal implementation**

Update the trainer to pass `mano_side`/`hand_is_right` into the hand GT batch.
Update `HandObjectLoss` to:
- select the correct MANO side for GT reconstruction
- keep the existing root-relative loss computation
- continue to use the predicted absolute vertices/joints from the model and subtract the root before comparing

Update visualization/debug helpers to import the vendored layer from `pi3.models.hamer` and use the same side-aware builder.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest debug/test_hand_object_loss.py debug/test_pi3x_dexycb_hand_integration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pi3/models/hand_object_loss.py trainers/pi3x_trainer.py pi3/visualization/pi3x_rerun_export.py debug/inspect_pi3x_pred_vs_gt.py debug/test_hand_object_loss.py
git commit -m "feat: align mano backend across training and inspection"
```

