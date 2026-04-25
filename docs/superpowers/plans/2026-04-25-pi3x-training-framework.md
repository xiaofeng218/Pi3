# Pi3X Hand/Object Training Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-stage Pi3X training framework that jointly optimizes hand and object outputs, freezes the scene backbone, and fine-tunes `ho_decoder` with LoRA.

**Architecture:** Keep the generic accelerator-based training skeleton, but add a dedicated `Pi3XTrainer`, a small local LoRA primitive, a centralized freeze policy, and a single `HandObjectLoss` integration point. `ho_decoder` will be adapted by injecting LoRA into the cross-attention projections first, while the rest of the scene stack stays frozen.

**Tech Stack:** PyTorch, HuggingFace Accelerate, Hydra, `unittest`, existing repo logging utilities, existing `HandObjectLoss`.

---

## File Map

- Create: `pi3/models/layers/lora.py` - local LoRA linear wrapper and module injection helpers for `HOBlockRope`.
- Modify: `pi3/models/layers/block.py` - accept LoRA configuration inside `HOBlockRope` and apply it to the chosen projections.
- Modify: `pi3/models/pi3x.py` - accept `ho_lora_cfg` / training-policy hooks and pass them into `HOBlockRope`.
- Create: `trainers/pi3x_training_policy.py` - centralized freeze / unfreeze policy for Pi3X training.
- Create: `trainers/pi3x_trainer.py` - dedicated trainer that wires model, loss, optimizer groups, logging, and validation.
- Modify: `trainers/__init__.py` - export `Pi3XTrainer`.
- Create: `configs/model/pi3x_hand_object.yaml` - Pi3X model config for this training mode.
- Create: `configs/loss/hand_object_loss.yaml` - `HandObjectLoss` config.
- Create: `configs/train/train_pi3x_hand_object.yaml` - optimizer / scheduler / freeze policy / logging config.
- Create: `configs/pi3x_hand_object.yaml` - top-level Hydra composition for the new training entry.
- Create: `debug/test_ho_decoder_lora.py` - LoRA unit test for `HOBlockRope`.
- Create: `debug/test_pi3x_training_policy.py` - freeze-policy test for trainable parameter selection.
- Create: `debug/test_pi3x_trainer_smoke.py` - trainer smoke test for optimizer grouping and loss wiring.

## Task 1: Add a local LoRA primitive and wire it into `HOBlockRope`

**Files:**
- Create: `pi3/models/layers/lora.py`
- Modify: `pi3/models/layers/block.py`
- Create: `debug/test_ho_decoder_lora.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from pi3.models.layers.block import HOBlockRope

def test_ho_block_lora_keeps_shape_and_freezes_base():
    blk = HOBlockRope(
        dim=64,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        init_values=0.01,
        qk_norm=True,
        rope=None,
        lora_cfg={"rank": 4, "alpha": 8, "targets": ["cross_attn"]},
    )
    x = torch.randn(2, 5, 64)
    y = torch.randn(2, 7, 64)
    out = blk(x, y, xpos=None, ypos=None)
    assert out.shape == x.shape
    trainable = [name for name, p in blk.named_parameters() if p.requires_grad]
    assert any("lora_" in name for name in trainable)
    assert all(
        ("lora_" in name) or ("norm" in name) or ("ls" in name) or name.endswith("bias")
        for name in trainable
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_ho_decoder_lora -v`

Expected: fail because `lora_cfg` and the LoRA wrapper do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# pi3/models/layers/lora.py
import math
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=8, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(cls, linear, rank, alpha):
        layer = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
        )
        layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x):
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_A), self.lora_B)
        return base + delta * self.scaling
```

```python
# pi3/models/layers/block.py
from .lora import LoRALinear

def _apply_lora_to_cross_attn(ho_blk, rank, alpha):
    ho_blk.cross_attn.q_proj = LoRALinear.from_linear(ho_blk.cross_attn.q_proj, rank=rank, alpha=alpha)
    ho_blk.cross_attn.k_proj = LoRALinear.from_linear(ho_blk.cross_attn.k_proj, rank=rank, alpha=alpha)
    ho_blk.cross_attn.v_proj = LoRALinear.from_linear(ho_blk.cross_attn.v_proj, rank=rank, alpha=alpha)
    ho_blk.cross_attn.proj = LoRALinear.from_linear(ho_blk.cross_attn.proj, rank=rank, alpha=alpha)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_ho_decoder_lora -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pi3/models/layers/lora.py pi3/models/layers/block.py debug/test_ho_decoder_lora.py
git commit -m "feat: add LoRA support for ho_decoder"
```

## Task 2: Add a centralized Pi3X freeze policy and thread LoRA config through model construction

**Files:**
- Create: `trainers/pi3x_training_policy.py`
- Modify: `pi3/models/pi3x.py`
- Modify: `trainers/__init__.py`
- Create: `debug/test_pi3x_training_policy.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from pi3.models.pi3x import Pi3X
from trainers.pi3x_training_policy import apply_pi3x_training_policy

def test_training_policy_only_leaves_expected_modules_trainable():
    model = Pi3X(use_multimodal=True, hand_mano_layer=None, ho_lora_cfg={"rank": 4, "alpha": 8})
    apply_pi3x_training_policy(model)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert any(name.startswith("ho_decoder") and "lora_" in name for name in trainable)
    assert "depth_emb" not in trainable
    assert "ray_embed.proj.weight" not in trainable
    assert "encoder.patch_embed.proj.weight" not in trainable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_pi3x_training_policy -v`

Expected: fail because `apply_pi3x_training_policy` and `ho_lora_cfg` plumbing do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# trainers/pi3x_training_policy.py
def apply_pi3x_training_policy(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if name.startswith(("hand_token_adapter", "object_query_adapter", "hand_mano_head", "object_pose_head", "register_token", "metric_token")):
            param.requires_grad = True
        if "ho_decoder" in name and ("lora_" in name or "norm" in name or "ls" in name or name.endswith("bias")):
            param.requires_grad = True
```

```python
# pi3/models/pi3x.py
def __init__(self, ckpt=None, use_multimodal=True, hamer_config_file=None, hamer_cache_dir=None, hand_mano_layer=None, ho_lora_cfg=None):
    super().__init__()
    self.ho_lora_cfg = ho_lora_cfg
    self.ho_decoder = nn.ModuleList([
        HOBlockRope(
            dim=1024,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            ffn_layer=Mlp,
            init_values=0.01,
            qk_norm=True,
            attn_class=FlashAttentionRope,
            rope=self.rope,
        ) for _ in range(36)
    ])
    if ho_lora_cfg is not None:
        for blk, ho_blk in zip(self.decoder, self.ho_decoder):
            init_ho_block_from_decoder_block(ho_blk, blk)
            ho_blk.enable_lora(**ho_lora_cfg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_pi3x_training_policy -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainers/pi3x_training_policy.py pi3/models/pi3x.py trainers/__init__.py debug/test_pi3x_training_policy.py
git commit -m "feat: add Pi3X training freeze policy"
```

## Task 3: Build `Pi3XTrainer` and connect `HandObjectLoss`

**Files:**
- Create: `trainers/pi3x_trainer.py`
- Modify: `trainers/__init__.py`
- Create: `debug/test_pi3x_trainer_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
from easydict import EasyDict
import torch
from trainers.pi3x_trainer import Pi3XTrainer

def test_pi3x_trainer_builds_optimizer_groups():
    cfg = EasyDict(
        train=EasyDict(
            optimizer=EasyDict(type="AdamW", lr=1e-4, weight_decay=0.05, betas=[0.9, 0.95], encoder_lr=1e-5),
            lr_scheduler=EasyDict(type="OneCycleLR", max_lr=1e-4, pct_start=0.0, anneal_strategy="cos", div_factor=1000.0, final_div_factor=100.0, total_steps=10),
            batch_size=1,
            num_epoch=1,
            iters_per_epoch=1,
            gradient_accumulation_steps=1,
            base_seed=666,
        ),
        loss=EasyDict(train_loss=EasyDict(_target_="pi3.models.hand_object_loss.HandObjectLoss")),
        model=EasyDict(_target_="pi3.models.pi3x.Pi3X"),
        log=EasyDict(output_dir="/tmp/pi3x-smoke", ckpt_dir="/tmp/pi3x-smoke/ckpts", ckpt_interval=1, max_checkpoints=1, summary_interval=1, use_tensorboard=False, use_wandb=False),
    )
    trainer = Pi3XTrainer.__new__(Pi3XTrainer)
    trainer.cfg = cfg
    groups = trainer.build_optimizer(cfg.train.optimizer, torch.nn.Linear(4, 4))
    assert len(groups) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_pi3x_trainer_smoke -v`

Expected: fail because `Pi3XTrainer` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# trainers/pi3x_trainer.py
from trainers.base_trainer_accelerate import BaseTrainer
from pi3.models.hand_object_loss import HandObjectLoss
from trainers.pi3x_training_policy import apply_pi3x_training_policy

class Pi3XTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_loss = HandObjectLoss(mano_layer=self.model.hand_mano_layer, **cfg.loss.train_loss)
        self.test_loss = HandObjectLoss(mano_layer=self.model.hand_mano_layer, **cfg.loss.train_loss)

    def prepare_model(self):
        model = super().prepare_model()
        apply_pi3x_training_policy(model)
        return model

    def build_optimizer(self, cfg_optimizer, model):
        def param_group_fn(model_):
            heads = []
            adapters = []
            lora = []
            for name, param in model_.named_parameters():
                if not param.requires_grad:
                    continue
                if "ho_decoder" in name and "lora_" in name:
                    lora.append(param)
                elif name.startswith(("hand_token_adapter", "object_query_adapter", "hand_mano_head", "object_pose_head", "register_token", "metric_token")):
                    adapters.append(param)
                else:
                    heads.append(param)
            return [
                {"params": heads, "lr": cfg_optimizer.lr, "weight_decay": cfg_optimizer.weight_decay},
                {"params": adapters, "lr": cfg_optimizer.lr, "weight_decay": 0.0},
                {"params": lora, "lr": cfg_optimizer.lr, "weight_decay": 0.0},
            ]
        return super().build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def forward_batch(self, batch, mode="train"):
        pred = self.model(**batch)
        return [pred, batch]

    def calculate_loss(self, output, batch, mode="train"):
        pred, batch = output
        loss, details = self.train_loss(pred, batch) if mode == "train" else self.test_loss(pred, batch)
        return EasyDict(loss=loss, **details)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_pi3x_trainer_smoke -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainers/pi3x_trainer.py trainers/__init__.py debug/test_pi3x_trainer_smoke.py
git commit -m "feat: add Pi3X trainer"
```

## Task 4: Add Hydra config composition, logging hooks, and a full smoke path

**Files:**
- Create: `configs/model/pi3x_hand_object.yaml`
- Create: `configs/loss/hand_object_loss.yaml`
- Create: `configs/train/train_pi3x_hand_object.yaml`
- Create: `configs/pi3x_hand_object.yaml`
- Modify: `trainers/pi3x_trainer.py`
- Create: `debug/test_pi3x_config_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from hydra import compose, initialize_config_dir

def test_pi3x_hand_object_config_composes():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, job_name="pi3x_smoke"):
        cfg = compose(config_name="pi3x_hand_object")
    assert cfg.trainer == "trainers.pi3x_trainer.Pi3XTrainer"
    assert cfg.model._target_ == "pi3.models.pi3x.Pi3X"
    assert cfg.loss.train_loss._target_ == "pi3.models.hand_object_loss.HandObjectLoss"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest debug.test_pi3x_config_smoke -v`

Expected: fail because the new config files do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```yaml
# configs/model/pi3x_hand_object.yaml
_target_: pi3.models.pi3x.Pi3X
ckpt: null
use_multimodal: true
hamer_config_file: null
hamer_cache_dir: null
hand_mano_layer: null
ho_lora_cfg:
  rank: 4
  alpha: 8
  targets: ["cross_attn"]
```

```yaml
# configs/loss/hand_object_loss.yaml
train_loss:
  _target_: pi3.models.hand_object_loss.HandObjectLoss
  root_index: 0
  geom_weight: 1.0
```

```yaml
# configs/pi3x_hand_object.yaml
defaults:
  - model: pi3x_hand_object
  - train: train_pi3x_hand_object
  - loss: hand_object_loss
  - data: dexycb.yaml
  - general: default.yaml
  - extras: default.yaml
  - override hydra/job_logging: custom
  - override hydra/hydra_logging: colorlog
  - _self_

trainer: trainers.pi3x_trainer.Pi3XTrainer
```

```yaml
# configs/train/train_pi3x_hand_object.yaml
train:
  num_epoch: 80
  iters_per_epoch: 800
  batch_size: 1
  gradient_accumulation_steps: 1
  optimizer:
    type: AdamW
    lr: 5e-5
    weight_decay: 5e-2
    betas: [0.9, 0.95]
    encoder_lr: 5e-6
  lr_scheduler:
    type: OneCycleLR
    max_lr: ${train.optimizer.lr}
    pct_start: 0.0
    anneal_strategy: cos
    div_factor: 1000.0
    final_div_factor: 100.0
    total_steps: -1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest debug.test_pi3x_config_smoke -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/model/pi3x_hand_object.yaml configs/loss/hand_object_loss.yaml configs/train/train_pi3x_hand_object.yaml configs/pi3x_hand_object.yaml trainers/pi3x_trainer.py debug/test_pi3x_config_smoke.py
git commit -m "feat: add Pi3X training configs"
```

## Coverage Check

- Spec section 5, batch/model/loss contract: covered by Task 3 and Task 4.
- Spec section 7, trainable parameter policy: covered by Task 1 and Task 2.
- Spec section 8, optimizer design: covered by Task 3.
- Spec section 9, forward / loss flow: covered by Task 3.
- Spec section 11, configuration layout: covered by Task 4.
- Spec section 12, trainer design: covered by Task 3 and Task 4.
- Spec section 13, logging / visualization: should be implemented inside `Pi3XTrainer` in Task 3, with the smoke test in Task 4 proving config wiring.
- Spec section 14, checkpoint / resume: inherited from `BaseTrainer`, verified implicitly once Task 3 and Task 4 are in place; if resume bugs show up, add a dedicated smoke test before merge.

## Execution Notes

- Keep the old `Pi3Trainer` path untouched.
- Do not add `peft` or `loralib` unless the local LoRA primitive proves insufficient.
- Keep the first LoRA surface limited to `HOBlockRope` cross-attention projections before expanding to self-attention or MLP.
- Save each task as a small commit so failures stay local and reviewable.
