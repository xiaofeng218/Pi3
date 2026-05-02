# Canonical Object Views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a DexYCB toolkit utility that renders 8 fixed canonical RGB and depth views for each object model, saving DexYCB-style files plus camera parameters referenced to the canonical object coordinate frame.

**Architecture:** Add one focused module under `dex_ycb_toolkit` that loads a model, normalizes it into a canonical object frame, constructs 8 fixed cube-corner views, renders textured RGB and canonical depth at `224x224`, and saves DexYCB-style outputs. Keep camera math, file IO, and rendering separated so tests can validate view ordering and metadata without requiring a live renderer.

**Tech Stack:** Python, numpy, trimesh, pyrender, imageio/cv2-compatible image writing, unittest/pytest.

---

### Task 1: Freeze camera and output contracts

**Files:**
- Create: `dex-ycb-toolkit/tests/test_object_canonical_views.py`
- Create: `dex-ycb-toolkit/dex_ycb_toolkit/object_canonical_views.py`

- [ ] **Step 1: Write the failing test**

```python
def test_generate_canonical_view_specs_uses_fixed_order_and_object_frame():
    specs = MODULE.generate_canonical_view_specs(radius=1.5)

    assert [spec.name for spec in specs] == [
        "nnn", "nnp", "npn", "npp", "pnn", "pnp", "ppn", "ppp",
    ]
    np.testing.assert_allclose(specs[0].target, np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(specs[-1].target, np.zeros(3, dtype=np.float32))
    assert specs[0].eye.shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dex-ycb-toolkit/tests/test_object_canonical_views.py::test_generate_canonical_view_specs_uses_fixed_order_and_object_frame -v`
Expected: FAIL because `object_canonical_views.py` or `generate_canonical_view_specs` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
VIEW_SIGNS = [
    (-1, -1, -1, "nnn"),
    (-1, -1, +1, "nnp"),
    (-1, +1, -1, "npn"),
    (-1, +1, +1, "npp"),
    (+1, -1, -1, "pnn"),
    (+1, -1, +1, "pnp"),
    (+1, +1, -1, "ppn"),
    (+1, +1, +1, "ppp"),
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dex-ycb-toolkit/tests/test_object_canonical_views.py::test_generate_canonical_view_specs_uses_fixed_order_and_object_frame -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dex-ycb-toolkit/tests/test_object_canonical_views.py dex-ycb-toolkit/dex_ycb_toolkit/object_canonical_views.py
git commit -m "feat: define canonical object view contracts"
```

### Task 2: Add normalization and camera serialization tests

**Files:**
- Modify: `dex-ycb-toolkit/tests/test_object_canonical_views.py`
- Modify: `dex-ycb-toolkit/dex_ycb_toolkit/object_canonical_views.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_normalize_vertices_maps_max_extent_to_unit_bbox():
    vertices = np.array([
        [1.0, 2.0, 3.0],
        [5.0, 4.0, 7.0],
    ], dtype=np.float32)

    normalized, center, scale = MODULE.normalize_vertices_to_unit_bbox(vertices)

    np.testing.assert_allclose(center, np.array([3.0, 3.0, 5.0], dtype=np.float32))
    assert scale == 4.0
    np.testing.assert_allclose(normalized.min(axis=0), np.array([-0.5, -0.25, -0.5], dtype=np.float32))
    np.testing.assert_allclose(normalized.max(axis=0), np.array([0.5, 0.25, 0.5], dtype=np.float32))

def test_save_render_bundle_uses_dexycb_names_and_persists_camera_params(tmp_path):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest dex-ycb-toolkit/tests/test_object_canonical_views.py -k "normalize_vertices or save_render_bundle" -v`
Expected: FAIL because the normalization and save helpers are not implemented.

- [ ] **Step 3: Write minimal implementation**

```python
def normalize_vertices_to_unit_bbox(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    scale = float(np.max(bbox_max - bbox_min))
    return (vertices - center) / scale, center.astype(np.float32), scale
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest dex-ycb-toolkit/tests/test_object_canonical_views.py -k "normalize_vertices or save_render_bundle" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dex-ycb-toolkit/tests/test_object_canonical_views.py dex-ycb-toolkit/dex_ycb_toolkit/object_canonical_views.py
git commit -m "feat: add canonical normalization and bundle saving"
```

### Task 3: Add render orchestration and CLI coverage

**Files:**
- Modify: `dex-ycb-toolkit/tests/test_object_canonical_views.py`
- Modify: `dex-ycb-toolkit/dex_ycb_toolkit/object_canonical_views.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_generate_object_views_wires_loader_renderer_and_bundle_writer(tmp_path):
    ...

def test_main_generates_one_named_object(tmp_path, monkeypatch):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest dex-ycb-toolkit/tests/test_object_canonical_views.py -k "generate_object_views or main_generates" -v`
Expected: FAIL because the orchestration and CLI entry points do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def generate_object_views(...):
    mesh = load_object_mesh(...)
    normalized = ...
    specs = ...
    rendered = renderer(...)
    save_render_bundle(...)

def main():
    args = parse_args()
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest dex-ycb-toolkit/tests/test_object_canonical_views.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dex-ycb-toolkit/tests/test_object_canonical_views.py dex-ycb-toolkit/dex_ycb_toolkit/object_canonical_views.py
git commit -m "feat: add canonical object view generation CLI"
```

### Task 4: Verify end-to-end behavior on the toolkit test suite

**Files:**
- Test: `dex-ycb-toolkit/tests/test_object_canonical_views.py`
- Test: `dex-ycb-toolkit/tests/test_prepare_sequence_for_vis_upload.py`
- Test: `dex-ycb-toolkit/tests/test_headless_vis.py`

- [ ] **Step 1: Run the focused tests**

Run: `pytest dex-ycb-toolkit/tests/test_object_canonical_views.py dex-ycb-toolkit/tests/test_prepare_sequence_for_vis_upload.py dex-ycb-toolkit/tests/test_headless_vis.py -v`
Expected: PASS

- [ ] **Step 2: Manually inspect one generated object bundle if rendering dependencies are available**

Run: `python -m dex_ycb_toolkit.object_canonical_views --dataset-root /data/hanxiaofeng/dataset/dexycb --object-name 025_mug --image-size 224 --output-subdir canonical_views_224`
Expected: writes eight `color_*.jpg`, eight `aligned_depth_to_color_*.png`, `camera_params.npz`, and `meta.json` under `models/025_mug/canonical_views_224/`.

- [ ] **Step 3: Commit**

```bash
git add dex-ycb-toolkit/tests/test_object_canonical_views.py dex-ycb-toolkit/dex_ycb_toolkit/object_canonical_views.py dex-ycb-toolkit/docs/superpowers/plans/2026-04-21-canonical-object-views.md
git commit -m "feat: generate canonical DexYCB object views"
```
