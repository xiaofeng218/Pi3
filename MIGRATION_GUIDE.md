# Pi3-training 项目迁移指南

本文档用于把当前项目迁移到另一台 Linux 服务器，并保证以下内容可用：

- 项目代码，包括顶层 `*.sh` 和 `dex-ycb-toolkit` 下的 `*.sh`
- conda 环境 `pi3`
- 所有已使用的模型权重、训练 checkpoint、DexYCB 数据

当前本机关键路径：

| 内容 | 当前路径 | 备注 |
| --- | --- | --- |
| 项目代码 | `/home/hanxiaofeng/Pi3-training` | `run_*.sh`、`train*.sh` 在这里 |
| conda 环境 | `/data/hanxiaofeng/miniconda3/envs/pi3` | Python 3.10.20，PyTorch 2.5.1+cu124 |
| DexYCB 数据 | `/data/hanxiaofeng/dataset/dexycb` | 约 67G |
| HaMeR 权重/cache | `/data/hanxiaofeng/models/hamer/_DATA` | 约 3.2G |
| Pi3X HF cache | `/data/hanxiaofeng/hub/models--yyfz233--Pi3X` | 约 5.1G |
| 训练输出/checkpoint | `/data/hanxiaofeng/models/pi3x-hoi` | 约 1.5G；项目内 `checkpoint` 是指向此目录的软链接 |

建议目标服务器优先复刻同样的 `/home/hanxiaofeng/Pi3-training` 和 `/data/hanxiaofeng/...` 路径。这样顶层脚本和 `configs/debug.yaml`、`configs/overfit.yaml` 里的硬编码路径不需要改。

## 0. 变量约定

下面命令默认在源服务器执行，除非特别标注。

```bash
export SRC_PROJECT=/home/hanxiaofeng/Pi3-training
export SRC_DATA_ROOT=/data/hanxiaofeng

export DST_HOST=<目标服务器用户名@目标服务器IP或域名>
export DST_PROJECT=/home/hanxiaofeng/Pi3-training
export DST_DATA_ROOT=/data/hanxiaofeng
```

如果目标服务器用户名或磁盘路径不同，把 `DST_PROJECT` 和 `DST_DATA_ROOT` 改成目标实际路径，并在第 6 节设置环境变量或修复硬编码路径。

## 1. 源服务器迁移前检查

```bash
cd "$SRC_PROJECT"

git status --short
find . -maxdepth 3 -name '*.sh' -print
conda env list
conda run -n pi3 python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)"

du -sh \
  "$SRC_PROJECT" \
  "$SRC_DATA_ROOT/dataset/dexycb" \
  "$SRC_DATA_ROOT/models/hamer/_DATA" \
  "$SRC_DATA_ROOT/hub/models--yyfz233--Pi3X" \
  "$SRC_DATA_ROOT/models/pi3x-hoi"
```

如果有未提交代码也需要迁移，直接用 `rsync` 迁移整个工作区即可，不依赖 git commit。

## 2. 打包 conda 环境 pi3

推荐用 `conda-pack`，因为它能最大程度保留当前环境中 conda 和 pip 包的实际状态。

```bash
conda install -n base -c conda-forge conda-pack
mkdir -p "$SRC_DATA_ROOT/migration_bundle"
conda pack -n pi3 -o "$SRC_DATA_ROOT/migration_bundle/pi3-conda-env.tar.gz"
```

同时导出可读清单，便于排查：

```bash
conda env export -n pi3 > "$SRC_DATA_ROOT/migration_bundle/pi3-environment-full.yml"
conda list -n pi3 --explicit > "$SRC_DATA_ROOT/migration_bundle/pi3-conda-explicit.txt"
conda run -n pi3 python -m pip freeze > "$SRC_DATA_ROOT/migration_bundle/pi3-pip-freeze.txt"
```

如果不能安装 `conda-pack`，备用方案是只传 `pi3-environment-full.yml` 和 `pi3-pip-freeze.txt`，在目标服务器重建环境；但完全一致性不如 `conda-pack`。

## 3. 生成迁移清单和校验文件

```bash
mkdir -p "$SRC_DATA_ROOT/migration_bundle"

cd "$SRC_PROJECT"
find . -path './.git' -prune -o -type f -print | sort > "$SRC_DATA_ROOT/migration_bundle/project-files.txt"

find \
  "$SRC_DATA_ROOT/models/hamer/_DATA" \
  "$SRC_DATA_ROOT/hub/models--yyfz233--Pi3X" \
  "$SRC_DATA_ROOT/models/pi3x-hoi" \
  -type f | sort > "$SRC_DATA_ROOT/migration_bundle/model-files.txt"

find "$SRC_DATA_ROOT/dataset/dexycb" -type f | sort > "$SRC_DATA_ROOT/migration_bundle/dexycb-files.txt"

sha256sum \
  "$SRC_DATA_ROOT/models/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt" \
  "$SRC_DATA_ROOT/models/hamer/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth" \
  > "$SRC_DATA_ROOT/migration_bundle/key-weights.sha256"
```

如果 Pi3X cache 中有 safetensors 文件，也可以追加校验：

```bash
find "$SRC_DATA_ROOT/hub/models--yyfz233--Pi3X" -type f -size +100M -print0 \
  | xargs -0 sha256sum >> "$SRC_DATA_ROOT/migration_bundle/key-weights.sha256"
```

## 4. 传输代码、环境、权重和数据

先在目标服务器创建目录：

```bash
ssh "$DST_HOST" "mkdir -p '$DST_PROJECT' '$DST_DATA_ROOT/migration_bundle' '$DST_DATA_ROOT/models/hamer' '$DST_DATA_ROOT/hub' '$DST_DATA_ROOT/models' '$DST_DATA_ROOT/dataset'"
```

传项目代码：

```bash
rsync -aH --info=progress2 --partial \
  "$SRC_PROJECT/" \
  "$DST_HOST:$DST_PROJECT/"
```

传 conda 环境包和清单：

```bash
rsync -aH --info=progress2 --partial \
  "$SRC_DATA_ROOT/migration_bundle/" \
  "$DST_HOST:$DST_DATA_ROOT/migration_bundle/"
```

传模型权重和训练 checkpoint：

```bash
rsync -aH --info=progress2 --partial \
  "$SRC_DATA_ROOT/models/hamer/_DATA/" \
  "$DST_HOST:$DST_DATA_ROOT/models/hamer/_DATA/"

rsync -aH --info=progress2 --partial \
  "$SRC_DATA_ROOT/hub/models--yyfz233--Pi3X/" \
  "$DST_HOST:$DST_DATA_ROOT/hub/models--yyfz233--Pi3X/"

rsync -aH --info=progress2 --partial \
  "$SRC_DATA_ROOT/models/pi3x-hoi/" \
  "$DST_HOST:$DST_DATA_ROOT/models/pi3x-hoi/"
```

传 DexYCB 数据：

```bash
rsync -aH --info=progress2 --partial \
  "$SRC_DATA_ROOT/dataset/dexycb/" \
  "$DST_HOST:$DST_DATA_ROOT/dataset/dexycb/"
```

如果网络不稳定，重复执行同一条 `rsync` 命令即可断点续传。

## 5. 目标服务器恢复 conda 环境

以下命令在目标服务器执行。

```bash
export DST_PROJECT=/home/hanxiaofeng/Pi3-training
export DST_DATA_ROOT=/data/hanxiaofeng
export CONDA_ROOT="$DST_DATA_ROOT/miniconda3"
```

如果目标服务器还没有 Miniconda，先安装：

```bash
mkdir -p "$DST_DATA_ROOT"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$CONDA_ROOT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda init bash
```

如果已经有 conda，只需要：

```bash
source "$CONDA_ROOT/etc/profile.d/conda.sh"
```

解压 `pi3` 环境：

```bash
mkdir -p "$CONDA_ROOT/envs/pi3"
tar -xzf "$DST_DATA_ROOT/migration_bundle/pi3-conda-env.tar.gz" -C "$CONDA_ROOT/envs/pi3"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate pi3
conda-unpack
```

验证环境：

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
python -c "import numpy, cv2, PIL, safetensors, huggingface_hub; print('basic imports ok')"
```

## 6. 目标服务器路径和环境变量

如果目标服务器复刻了 `/data/hanxiaofeng/...`，建议设置：

```bash
export DEXYCB_ROOT=/data/hanxiaofeng/dataset/dexycb
export DEX_YCB_DIR=/data/hanxiaofeng/dataset/dexycb
export HAMER_CONFIG_FILE=/data/hanxiaofeng/models/hamer/_DATA/hamer_ckpts/model_config.yaml
export HAMER_ENCODER_CKPT=/data/hanxiaofeng/models/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
export HAMER_CACHE_DIR=/data/hanxiaofeng/models/hamer/_DATA
export PI3X_CKPT=/data/hanxiaofeng/hub/models--yyfz233--Pi3X/snapshots/bb1deea4d7423de5b30691739cb451a3f57dc1d5
export HF_HOME=/data/hanxiaofeng/hub
```

可以把这些写入 `~/.bashrc` 或项目专用启动文件：

```bash
cat > "$DST_PROJECT/env.migration.sh" <<'EOF'
export DEXYCB_ROOT=/data/hanxiaofeng/dataset/dexycb
export DEX_YCB_DIR=/data/hanxiaofeng/dataset/dexycb
export HAMER_CONFIG_FILE=/data/hanxiaofeng/models/hamer/_DATA/hamer_ckpts/model_config.yaml
export HAMER_ENCODER_CKPT=/data/hanxiaofeng/models/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
export HAMER_CACHE_DIR=/data/hanxiaofeng/models/hamer/_DATA
export PI3X_CKPT=/data/hanxiaofeng/hub/models--yyfz233--Pi3X/snapshots/bb1deea4d7423de5b30691739cb451a3f57dc1d5
export HF_HOME=/data/hanxiaofeng/hub
export PYTHONPATH=/home/hanxiaofeng/Pi3-training:/home/hanxiaofeng/Pi3-training/dex-ycb-toolkit:/home/hanxiaofeng/Pi3-training/third_party/hamer:/home/hanxiaofeng/Pi3-training/third_party/manopth:${PYTHONPATH}
EOF
source "$DST_PROJECT/env.migration.sh"
```

如果目标路径不是 `/data/hanxiaofeng`，用 `sed` 修复硬编码路径：

```bash
cd "$DST_PROJECT"

OLD_PROJECT=/home/hanxiaofeng/Pi3-training
NEW_PROJECT="$DST_PROJECT"
OLD_DATA=/data/hanxiaofeng
NEW_DATA="$DST_DATA_ROOT"

grep -RIl "$OLD_PROJECT\|$OLD_DATA" \
  run_*.sh train*.sh configs debug dex-ycb-toolkit \
  | xargs -r sed -i \
    -e "s#${OLD_PROJECT}#${NEW_PROJECT}#g" \
    -e "s#${OLD_DATA}#${NEW_DATA}#g"
```

## 7. 恢复项目内软链接和权限

在目标服务器执行：

```bash
cd "$DST_PROJECT"

ln -sfn "$DST_DATA_ROOT/models/pi3x-hoi" checkpoint
chmod +x ./*.sh
find dex-ycb-toolkit -name '*.sh' -exec chmod +x {} \;
find third_party -name '*.sh' -exec chmod +x {} \;
```

`vis_upload` 当前是项目内软链接，源仓库里指向 `../vis_upload`。如果目标服务器没有该目录，而你不需要上传可视化结果，可以先忽略；如果需要，额外迁移源服务器的 `/home/hanxiaofeng/vis_upload` 到目标同级目录。

## 8. 安装 editable 包

在目标服务器执行：

```bash
source "$DST_DATA_ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate pi3
cd "$DST_PROJECT"

python -m pip install -e .
python -m pip install -e third_party/manopth
python -m pip install -e third_party/hamer
python -m pip install -e dex-ycb-toolkit
```

如果 `python -m pip install -e .` 因为项目根目录没有 `setup.py` 或 `pyproject.toml` 失败，可以忽略根目录这一步，保留 `PYTHONPATH` 即可。

`dex-ycb-toolkit/run_obj_render.sh` 使用 EGL/OpenGL。目标服务器如果缺库，安装系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
```

## 9. 校验迁移完整性

在目标服务器执行：

```bash
cd "$DST_PROJECT"
source "$DST_DATA_ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate pi3
source "$DST_PROJECT/env.migration.sh"

bash -n ./*.sh
find dex-ycb-toolkit -name '*.sh' -print0 | xargs -0 -n1 bash -n

test -d "$DEXYCB_ROOT"
test -f "$HAMER_CONFIG_FILE"
test -f "$HAMER_ENCODER_CKPT"
test -d "$PI3X_CKPT"

sha256sum -c "$DST_DATA_ROOT/migration_bundle/key-weights.sha256"
```

如果目标服务器没有复刻 `/data/hanxiaofeng`，先把校验文件里的源路径替换成目标路径：

```bash
sed -i "s#/data/hanxiaofeng#${DST_DATA_ROOT}#g" "$DST_DATA_ROOT/migration_bundle/key-weights.sha256"
sha256sum -c "$DST_DATA_ROOT/migration_bundle/key-weights.sha256"
```

基础 Python import：

```bash
python - <<'PY'
import torch
from pi3.models.pi3x import Pi3X
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("imports ok")
PY
```

## 10. 验证顶层脚本

先做语法检查：

```bash
cd "$DST_PROJECT"
bash -n run_debug.sh
bash -n run_origin_demo.sh
bash -n train.sh
bash -n train_overfit.sh
```

再按需执行。建议先跑轻量脚本：

```bash
source "$DST_PROJECT/env.migration.sh"
bash run_origin_demo.sh
```

`run_debug.sh` 当前默认读取：

```text
/data/hanxiaofeng/dataset/dexycb
```

如果目标路径不同，请先完成第 6 节硬编码路径替换，或者手动修改 `run_debug.sh` 中的 `--data-root`。

训练 smoke test：

```bash
source "$DST_PROJECT/env.migration.sh"
bash train.sh
```

如果只想快速验证配置和单步训练，可以先执行：

```bash
python scripts/train_pi3x.py --config-name debug
```

## 11. 验证 dex-ycb-toolkit 脚本

在目标服务器执行：

```bash
cd "$DST_PROJECT/dex-ycb-toolkit"
source "$DST_PROJECT/env.migration.sh"

bash -n run_data_vis.sh
bash -n run_obj_render.sh
bash -n download_dexycb.sh
find . -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

数据可视化：

```bash
bash run_data_vis.sh
```

物体 canonical view 渲染：

```bash
bash run_obj_render.sh "$DEXYCB_ROOT" canonical_views_224 45 224
```

`download_dexycb.sh` 目前写死：

```bash
cd /data/hanxiaofeng/dataset/dexycb
```

如果目标路径不同，先替换脚本中的 `/data/hanxiaofeng/dataset/dexycb`，或者手动创建同路径软链接：

```bash
sudo mkdir -p /data/hanxiaofeng/dataset
sudo ln -sfn "$DEXYCB_ROOT" /data/hanxiaofeng/dataset/dexycb
```

## 12. 常见问题

### 找不到 Pi3X 权重

确认：

```bash
ls -lah "$PI3X_CKPT"
find "$DST_DATA_ROOT/hub/models--yyfz233--Pi3X" -maxdepth 5 -type f
```

如果只迁移了 HuggingFace cache 但没有设置 `PI3X_CKPT`，`Pi3X.from_pretrained("yyfz233/Pi3X")` 可能尝试联网。离线环境建议始终设置 `PI3X_CKPT` 到本地 snapshot 目录。

### 找不到 HaMeR 权重

确认：

```bash
ls -lah "$HAMER_CONFIG_FILE"
ls -lah "$HAMER_ENCODER_CKPT"
ls -lah "$HAMER_CACHE_DIR/vitpose_ckpts/vitpose+_huge/wholebody.pth"
```

### `ModuleNotFoundError`

先确认 `PYTHONPATH`：

```bash
echo "$PYTHONPATH"
python -c "import pi3, dex_ycb_toolkit; print('ok')"
```

如果失败，重新执行第 6 节和第 8 节。

### CUDA 不可用

当前环境是 `torch 2.5.1+cu124`。目标服务器需要 NVIDIA 驱动支持 CUDA 12.x 运行时：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

如果 `nvidia-smi` 不可用，需要先安装或修复目标服务器 NVIDIA 驱动。

## 13. 最小通过标准

迁移完成后至少满足：

```bash
cd "$DST_PROJECT"
source "$DST_DATA_ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate pi3
source "$DST_PROJECT/env.migration.sh"

bash -n ./*.sh
find dex-ycb-toolkit -name '*.sh' -print0 | xargs -0 -n1 bash -n
python -c "import torch; import pi3; import dex_ycb_toolkit; print('imports ok')"
test -d "$DEXYCB_ROOT"
test -f "$HAMER_ENCODER_CKPT"
test -d "$PI3X_CKPT"
```

这些通过后，再运行：

```bash
bash run_origin_demo.sh
bash dex-ycb-toolkit/run_data_vis.sh
python scripts/train_pi3x.py --config-name debug
```
