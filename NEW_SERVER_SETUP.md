# 新服务器配置与资产下载指南

本文档用于在另一台服务器从 GitHub 拉取本仓库，并标准化下载训练需要的数据和模型权重。

## 1. 克隆代码

```bash
git clone https://github.com/xiaofeng218/Pi3.git
cd Pi3
```

建议后续所有命令都在仓库根目录执行。

## 2. 创建 conda 环境

仓库内提供了当前可用环境的导出文件：[conda/pi3-environment.yml](/home/hanxiaofeng/Pi3-training/conda/pi3-environment.yml)。

推荐直接创建：

```bash
conda env create -f conda/pi3-environment.yml
conda activate pi3
pip install -e dex-ycb-toolkit
```

如果环境已存在，使用更新命令：

```bash
conda env update -n pi3 -f conda/pi3-environment.yml --prune
conda activate pi3
pip install -e dex-ycb-toolkit
```

`dex-ycb-toolkit/run_obj_render.sh` 需要 EGL/OpenGL，缺库时安装：

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
```

说明：

- 该环境文件固定了当前机器验证通过的 Python、PyTorch CUDA 12.4 和相关 pip 依赖版本。
- 文件里已移除本机 `prefix`，可以直接在新服务器使用。
- `dex-ycb-toolkit` 仍建议在建好环境后执行 `pip install -e dex-ycb-toolkit`，因为它依赖当前仓库代码本身，而不是单独从 PyPI 安装。

## 3. 标准资产目录

本仓库约定所有外部资产放在 `PI3_DATA_ROOT` 下。默认值是仓库根目录的 `data/`：

```text
data/
  model/
    pi3x/Pi3X/
    hamer/_DATA/
  dataset/
    dexycb/
```

这些目录由 `.gitignore` 屏蔽，不提交到 GitHub。

如果希望把数据和权重放到项目外的大磁盘，例如 `/data/pi3-assets`，先设置：

```bash
export PI3_DATA_ROOT=/data/pi3-assets
```

然后再加载标准环境变量。

加载标准环境变量：

```bash
source asset_registry/env.sh
```

关键变量：

```bash
echo "$PI3_MODEL_ROOT"       # data/model
echo "$PI3_DATASET_ROOT"     # data/dataset
echo "$PI3X_CKPT"            # data/model/pi3x/Pi3X
echo "$HAMER_CACHE_DIR"      # data/model/hamer/_DATA
echo "$HAMER_ENCODER_CKPT"   # data/model/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
echo "$DEXYCB_ROOT"          # data/dataset/dexycb
```

设置 `PI3_DATA_ROOT=/data/pi3-assets` 后，上述路径会自动变为：

```text
/data/pi3-assets/model/...
/data/pi3-assets/dataset/...
```

## 4. 下载模型权重

当前训练链路需要：

- Pi3X：`yyfz233/Pi3X`
- HaMeR：`hamer_demo_data.tar.gz`

执行：

```bash
conda activate pi3
source asset_registry/env.sh
bash asset_registry/model/download_models.sh
```

下载完成后应看到：

```text
data/model/pi3x/Pi3X/
data/model/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
data/model/hamer/_DATA/hamer_ckpts/model_config.yaml
data/model/hamer/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth
data/model/hamer/_DATA/data/mano/MANO_RIGHT.pkl
data/model/hamer/_DATA/data/mano/MANO_LEFT.pkl
data/model/hamer/_DATA/data/mano_mean_params.npz
```

注意：当前 `scripts/train_pi3x.py` 不需要单独下载 Pi3 权重。

## 5. 下载 DexYCB 数据

执行：

```bash
conda activate pi3
export PI3_DATA_ROOT=/data/pi3-assets   # 可选；需与模型下载时保持一致
source asset_registry/env.sh
bash asset_registry/dataset/download_datasets.sh
```

该脚本会调用：

```bash
dex-ycb-toolkit/download_dexycb.sh "$DEXYCB_ROOT"
```

默认写入：

```text
data/dataset/dexycb/
```

## 6. 生成 DexYCB Object Canonical Views

训练配置使用：

```yaml
object_multiview_subdir: canonical_views_224
```

因此 DexYCB 下载完成后，需要生成物体 canonical views：

```bash
conda activate pi3
source asset_registry/env.sh
bash dex-ycb-toolkit/run_obj_render.sh "$DEXYCB_ROOT" canonical_views_224 45 224
```

输出会写入每个 YCB object model 目录下的 `canonical_views_224/`。

## 7. 配置检查

```bash
conda activate pi3
source asset_registry/env.sh

test -d "$PI3X_CKPT"
test -f "$HAMER_ENCODER_CKPT"
test -d "$DEXYCB_ROOT"
test -f "$DEXYCB_ROOT/20200709-subject-01/20200709_141754/meta.yml"

python -c "import torch; import pi3; import dex_ycb_toolkit; print('imports ok', torch.__version__)"
```

检查 Hydra 配置：

```bash
python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir

config_dir = str(Path.cwd() / "configs")
with initialize_config_dir(config_dir=config_dir, job_name="check", version_base=None):
    cfg = compose(config_name="pi3x_hand_object")

print("data_root:", cfg.train_dataset.data_root)
print("pi3x ckpt:", cfg.model.ckpt)
print("hamer cache:", cfg.model.hamer_cache_dir)
print("hand encoder ckpt:", cfg.hand_encoder.checkpoint)
PY
```

## 8. 运行 smoke test

轻量配置：

```bash
conda activate pi3
source asset_registry/env.sh
python scripts/train_pi3x.py --config-name debug
```

标准入口：

```bash
bash train.sh
```

过拟合调试：

```bash
bash train_overfit.sh
```

调试可视化：

```bash
bash run_debug.sh
```

## 9. 常见问题

### 找不到 Pi3X 权重

确认：

```bash
source asset_registry/env.sh
ls -lah "$PI3X_CKPT"
```

如果目录不存在，重新执行：

```bash
bash asset_registry/model/download_models.sh
```

### 找不到 HaMeR / MANO 文件

确认：

```bash
source asset_registry/env.sh
ls -lah "$HAMER_ENCODER_CKPT"
ls -lah "$HAMER_CACHE_DIR/data/mano/MANO_RIGHT.pkl"
ls -lah "$HAMER_CACHE_DIR/data/mano/MANO_LEFT.pkl"
ls -lah "$HAMER_CACHE_DIR/data/mano_mean_params.npz"
```

### 找不到 DexYCB 数据

确认：

```bash
source asset_registry/env.sh
ls -lah "$DEXYCB_ROOT"
```

如果下载中断，重复执行：

```bash
bash asset_registry/dataset/download_datasets.sh
```

### CUDA 不可用

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

如果 `torch.cuda.is_available()` 为 `False`，优先检查 NVIDIA 驱动和 PyTorch CUDA wheel 是否匹配。
