"""
Diagnose why template_vertices is missing from object_multiview.

Run: python debug/test_object_multiview_keys.py
"""
from __future__ import annotations

import sys, os, importlib
sys.path.insert(0, ".")

import datasets.dexycb_dataset
importlib.reload(datasets.dexycb_dataset)
from datasets.dexycb_dataset import DexYCBDataset

data_root = os.environ.get("DEXYCB_ROOT",
    os.environ.get("PI3_DATA_ROOT", "data") + "/dataset/dexycb")
print(f"data_root = {data_root}")

ds_kwargs = dict(
    data_root=data_root,
    split_style="s0_like_subject01",
    subject="20200709-subject-01",
    object_multiview_subdir="canonical_views_224",
    use_crop=False,
    mode="train",
    z_far=0,
    frame_num=8,
    vertex_sample_count=2048,
    resolution=[[224, 224]],
    shuffle=True,
)

# ============ TEST 1: Direct instantiation ============
print("=" * 60)
print("TEST 1: Direct dataset (no workers)")
ds = DexYCBDataset(**ds_kwargs)
print(f"object_multiview_subdir = {ds.object_multiview_subdir!r}")
print(f"vertex_sample_count = {ds.vertex_sample_count}")

sample = ds[0]
v0 = sample[0]
om = v0["object_multiview"]
has = "template_vertices" in om
print(f"view[0] object_multiview keys = {sorted(om.keys())}")
print(f"template_vertices present: {has}")
if has:
    print(f"  shape: {om['template_vertices'].shape}")
else:
    print("  MISSING! Debugging _prepare_object_multiview...")
    track = ds.tracks[0]
    result = ds._prepare_object_multiview(track["grasped_object_id"])
    print(f"  _prepare_object_multiview result type: {type(result).__name__}")
    if result is not None:
        print(f"  _prepare_object_multiview keys: {sorted(result.keys())}")
    else:
        print(f"  _prepare_object_multiview returned None!")
        print(f"  _load_object_multiview_bundle result: {ds._load_object_multiview_bundle(track['grasped_object_id'])}")

# ============ TEST 2: DataLoader workers=1 ============
print()
print("=" * 60)
print("TEST 2: DataLoader with num_workers=1")

from torch.utils.data import DataLoader
from datasets.base.utils import unified_collate_fn

# Need to use batch_sampler like the training code does
from datasets.base.batched_sampler import DynamicBatchSampler, DynamicDistributedSampler

ds2 = DexYCBDataset(**ds_kwargs)
sampler = DynamicDistributedSampler(ds2, num_replicas=1, rank=0, seed=666,
                                     shuffle=True, drop_last=True)
batch_sampler = DynamicBatchSampler(sampler, num_resolution=1,
                                     image_num_range=[12, 12], seed=666,
                                     max_img_per_gpu=24, rank=0)
dl = DataLoader(dataset=ds2, batch_sampler=batch_sampler, num_workers=1,
                pin_memory=False, collate_fn=unified_collate_fn,
                persistent_workers=False, prefetch_factor=1)

import torch
batch = next(iter(dl))
print(f"batch has {len(batch)} views")
b0 = batch[0]
om0 = b0["object_multiview"]
if isinstance(om0, dict):
    keys0 = sorted(om0.keys())
    has0 = "template_vertices" in om0
    print(f"view[0] object_multiview keys = {keys0}")
    print(f"template_vertices present: {has0}")
else:
    print(f"view[0] object_multiview is {type(om0).__name__}, not dict!")

# Check all views
for i, view in enumerate(batch):
    v_om = view["object_multiview"]
    if isinstance(v_om, dict):
        print(f"  view[{i}]: template_vertices={'YES' if 'template_vertices' in v_om else 'NO '}  "
              f"keys={len(v_om.keys())}")
    else:
        print(f"  view[{i}]: not a dict ({type(v_om).__name__})")

# ============ TEST 3: DataLoader workers=0 ============
print()
print("=" * 60)
print("TEST 3: DataLoader with num_workers=0")

ds3 = DexYCBDataset(**ds_kwargs)
sampler3 = DynamicDistributedSampler(ds3, num_replicas=1, rank=0, seed=666,
                                      shuffle=True, drop_last=True)
batch_sampler3 = DynamicBatchSampler(sampler3, num_resolution=1,
                                      image_num_range=[12, 12], seed=666,
                                      max_img_per_gpu=24, rank=0)
dl3 = DataLoader(dataset=ds3, batch_sampler=batch_sampler3, num_workers=0,
                 pin_memory=False, collate_fn=unified_collate_fn)

batch3 = next(iter(dl3))
b03 = batch3[0]
om03 = b03["object_multiview"]
if isinstance(om03, dict):
    print(f"view[0] object_multiview keys = {sorted(om03.keys())}")
    print(f"template_vertices present: {'template_vertices' in om03}")
else:
    print(f"view[0] object_multiview is {type(om03).__name__}")

print()
print("=" * 60)
print("CONCLUSION: if TEST 1/3 passes but TEST 2 fails, the issue is")
print("in the DataLoader worker's import of the dataset module.")
