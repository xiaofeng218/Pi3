# HaMeR Mask Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new HaMeR demo that reads one DexYCB RGB image and its hand mask, derives the hand bounding box directly from the mask, and runs HaMeR without detectron2 or ViTPose.

**Architecture:** Reuse the existing HaMeR inference path from `demo.py` after the hand-box stage. Add a small helper to export one DexYCB example RGB image plus a binary hand mask into `third_party/hamer/example_data`, then build a new demo script that computes the bbox from that mask and feeds the crop into `ViTDetDataset` and HaMeR.

**Tech Stack:** Python, NumPy, OpenCV, PIL, PyTorch, HaMeR local modules.

---

### Task 1: Add failing tests for mask-derived bbox helpers

**Files:**
- Create: `third_party/hamer/tests/test_demo_mask_bbox_dexycb.py`
- Test: `third_party/hamer/tests/test_demo_mask_bbox_dexycb.py`

- [ ] **Step 1: Write the failing tests**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement minimal helper functions**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Add DexYCB example export helper and mask-based demo

**Files:**
- Create: `third_party/hamer/prepare_dexycb_example_data.py`
- Create: `third_party/hamer/demo_mask_bbox_dexycb.py`
- Modify: `third_party/hamer/tests/test_demo_mask_bbox_dexycb.py`

- [ ] **Step 1: Add export helper for one fixed DexYCB RGB+mask example**
- [ ] **Step 2: Add mask-bbox HaMeR demo using exported example files by default**
- [ ] **Step 3: Add/extend tests for export helper behavior**
- [ ] **Step 4: Run tests**

### Task 3: Verify with real data in the `pi3` environment

**Files:**
- Modify: `third_party/hamer/example_data/*` via the export helper at runtime

- [ ] **Step 1: Run the export helper against `/data/hanxiaofeng/dataset/dexycb`**
- [ ] **Step 2: Run the new demo once**
- [ ] **Step 3: Confirm output artifacts land in `third_party/hamer/out_demo*` or the chosen folder**
