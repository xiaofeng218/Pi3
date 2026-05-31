# Pi3 云端迁移指南

本文档记录如何把当前项目的代码、训练数据和模型权重迁移到云端（GitHub + Cloudflare R2），以及在新服务器上完整还原环境。

---

## 1. 资产清单

### 1.1 代码 → GitHub

| 内容 | 远程仓库 |
|------|---------|
| Pi3 项目代码 | `https://github.com/xiaofeng218/Pi3.git`，分支 `training` |
| ForeHOI 预处理代码 | 新建仓库（见第 6 节） |

### 1.2 模型权重 → Cloudflare R2

| 资产 | 当前路径 | 大小 | R2 路径 |
|------|---------|------|---------|
| HaMeR 权重 | `data/model/hamer/_DATA` | ~12G | `r2:pi3-models/hamer/` |
| Pi3X 权重 | `data/model/pi3x` | ~5.1G | `r2:pi3-models/pi3x/` |

### 1.3 ForeHOI 生成数据 → Cloudflare R2

以下是 pi3x-pipeline 生成的数据，**无法从网上重新下载**，必须上传到 R2。

| 目录 | 大小 | R2 路径 |
|------|------|---------|
| `ForeHOI/outputs/pi3x-pipeline/render/` | ~35G | `r2:pi3-forehoi/render/` |
| `ForeHOI/outputs/pi3x-pipeline/pi3x_raw/` | ~25M（含 symlink 指向 render） | `r2:pi3-forehoi/pi3x_raw/` |
| `ForeHOI/outputs/pi3x-pipeline/object_multiview_pyrender/` | ~182M | `r2:pi3-forehoi/object_multiview_pyrender/` |
| `ForeHOI/outputs/pi3x-pipeline/object_parts/` | ~2G | `r2:pi3-forehoi/object_parts/` |
| `ForeHOI/outputs/pi3x-pipeline/cache/` | ~5.8G | `r2:pi3-forehoi/cache/`（可选，节省重新生成时间） |

> **注意**：`pi3x_raw/` 下的 rgb/depth/mask 子目录是软链接，指向 `render/` 对应序列。上传时用 `--skip-links`，下载后用脚本重建。

### 1.4 ForeHOI 静态资产 → Cloudflare R2

以下数据虽来自公开来源，但提取/整理成本高，直接上传到 R2。

| 目录 | 大小 | R2 路径 | 原因 |
|------|------|---------|------|
| `ForeHOI/DART/texture&accessories/` | ~1.9G | `r2:pi3-forehoi-deps/dart-texture-accessories/` | HuggingFace 只提供含全部 DARTset 的大压缩包，无法单独下载此子目录 |

### 1.5 ForeHOI 原始依赖 → 需重新下载

以下数据来自公开来源，**不上传**到 R2，在新服务器直接重新下载。详见第 4 节。

| 目录 | 大小 | 来源 |
|------|------|------|
| `ForeHOI/graspxl_renders/` | ~7.9G | HuggingFace `YuantaoChen/ForeHOI` |
| `ForeHOI/front-3d/` | ~34G | 3D-FRONT（Alibaba，需邮件申请） |
| `ForeHOI/objaverse_assets/` | ~1.6G | Objaverse（Python 包自动下载） |
| `ForeHOI/blender-bin/` | ~2.3G | Blender 官网 |

### 1.6 不迁移

| 目录 | 原因 |
|------|------|
| DexYCB | 用 `asset_registry/dataset/download_datasets.sh` 重新下载 |
| Hot3D | 暂未使用 |
| `ForeHOI/samples/` | debug 产物，可再生成 |
| `ForeHOI/DART/`（除 texture&accessories） | 见 1.5，仅上传所需子目录 |
| conda 环境 | 用 `conda/pi3-environment.yml` 在新服务器重建 |

---

## 2. 工具准备：安装 rclone 并配置 R2

在**源服务器**和**新服务器**都执行：

```bash
curl https://rclone.org/install.sh | sudo bash
```

配置 R2 remote（执行一次，填入你的 Cloudflare R2 凭证）：

```bash
rclone config create r2 s3 \
  provider Cloudflare \
  access_key_id <R2_ACCESS_KEY_ID> \
  secret_access_key <R2_SECRET_ACCESS_KEY> \
  endpoint https://<CLOUDFLARE_ACCOUNT_ID>.r2.cloudflarestorage.com \
  acl private
```

凭证获取：Cloudflare Dashboard → R2 → Manage API Tokens → Create Token（权限选 Object Read & Write）。

验证：

```bash
rclone lsd r2:
```

---

## 3. 从当前服务器上传到 R2

以下命令在当前服务器执行。变量约定：

```bash
export FOREHOI_SRC=/mnt/2/data/dataset/ForeHOI/outputs/pi3x-pipeline
export MODEL_SRC=/mnt/2/projects/Pi3/data/model
```

### 3.1 上传模型权重

```bash
rclone copy "$MODEL_SRC/hamer/_DATA" r2:pi3-models/hamer \
  --progress --transfers 4 --checkers 8

rclone copy "$MODEL_SRC/pi3x" r2:pi3-models/pi3x \
  --progress --transfers 4
```

### 3.2 上传 ForeHOI 生成数据

上传 render（最大，35G，建议后台执行）：

```bash
rclone copy "$FOREHOI_SRC/render" r2:pi3-forehoi/render \
  --progress --transfers 8 --checkers 16 --multi-thread-streams 4
```

上传 pi3x_raw（跳过 symlink，只传 meta 文件）：

```bash
rclone copy "$FOREHOI_SRC/pi3x_raw" r2:pi3-forehoi/pi3x_raw \
  --skip-links \
  --progress --transfers 4
```

上传 object_multiview_pyrender：

```bash
rclone copy "$FOREHOI_SRC/object_multiview_pyrender" r2:pi3-forehoi/object_multiview_pyrender \
  --progress --transfers 8
```

上传 object_parts：

```bash
rclone copy "$FOREHOI_SRC/object_parts" r2:pi3-forehoi/object_parts \
  --progress --transfers 4
```

上传 cache（可选，5.8G，节省未来重新处理时间）：

```bash
rclone copy "$FOREHOI_SRC/cache" r2:pi3-forehoi/cache \
  --progress --transfers 4
```

### 3.3 上传 DART texture&accessories（1.9G）

```bash
export DART_SRC=/mnt/2/data/dataset/ForeHOI/DART

rclone copy "$DART_SRC/texture&accessories" r2:pi3-forehoi-deps/dart-texture-accessories \
  --progress --transfers 4
```

### 3.4 验证上传

```bash
rclone size r2:pi3-forehoi/render
rclone size r2:pi3-models/hamer
rclone size r2:pi3-forehoi-deps/dart-texture-accessories
```

---

## 4. ForeHOI 原始依赖下载方法

在**新服务器**上，把 ForeHOI 工作目录设为 `$FOREHOI_WORK_ROOT`（默认与生成数据分开存放）。

```bash
export FOREHOI_WORK_ROOT=/data/forehoi-deps   # 可按需调整
mkdir -p "$FOREHOI_WORK_ROOT"
```

### 4.1 graspxl_renders（7.9G）

来源：HuggingFace Dataset `YuantaoChen/ForeHOI`

```bash
pip install -U huggingface_hub

hf download YuantaoChen/ForeHOI \
  --repo-type dataset \
  --include "graspxl_renders/**/metadata.npy" \
  --include "graspxl_renders/**/object_mesh.obj" \
  --include "graspxl_renders/**/transforms.json" \
  --exclude "graspxl_renders/**/data.tar" \
  --local-dir "$FOREHOI_WORK_ROOT" \
  --max-workers 4
```

如在中国大陆网络，加环境变量：

```bash
HF_ENDPOINT=https://hf-mirror.com hf download YuantaoChen/ForeHOI ...（同上）
```

### 4.2 objaverse_assets（1.6G，868 个 .glb 文件）

来源：Objaverse（通过 Python 包按 UID 下载）

依赖 `graspxl_renders/` 已下载完毕，脚本从中提取所需 UID：

```bash
pip install objaverse

# 用 ForeHOI 自带脚本下载对应 Objaverse 资产
cd "$FOREHOI_WORK_ROOT"
python download_forehoi_minimal.py \
  --root-dir "$FOREHOI_WORK_ROOT" \
  --repo-id YuantaoChen/ForeHOI \
  --skip-forehoi \
  --failed-uids-json "$FOREHOI_WORK_ROOT/objaverse_failed_uids.json" \
  --failed-uids-txt  "$FOREHOI_WORK_ROOT/objaverse_failed_uids.txt"
```

下载完成后验证：

```bash
ls "$FOREHOI_WORK_ROOT/objaverse_assets/" | wc -l   # 应为 868
```

### 4.3 3D-FRONT（~34G）

来源：Alibaba Topping Homestyler，**需要申请许可证**。

申请步骤：
1. 下载并签署 [3D-FRONT 数据集使用协议](https://gw.alicdn.com/bao/uploaded/TB1ZJUfK.z1gK0jSZLeXXb9kVXa.pdf)
2. 发送签署后的 PDF 到 `tianchi_open_dataset@alibabacloud.com`，邮件注明姓名、导师/负责人姓名和所在机构
3. 收到回复后会获得带过期时间的下载 URL

收到 URL 后，将下载脚本中的 URL 替换为新获得的 URL，然后执行：

```bash
cd "$FOREHOI_WORK_ROOT/front-3d"
bash download.sh
```

需要下载的三个文件：
- `3D-FRONT.zip`（场景布局 JSON，~17G 解压后）
- `3D-FUTURE-model-part1.zip`（家具模型，~16G 解压后）
- `3D-FRONT-texture.zip`（贴图，~1.8G 解压后）

解压到同一目录：

```bash
cd "$FOREHOI_WORK_ROOT/front-3d"
unzip 3D-FRONT.zip
unzip 3D-FUTURE-model-part1.zip
unzip 3D-FRONT-texture.zip
```

### 4.4 DART texture&accessories（1.9G）

来源：Cloudflare R2（已上传，见 3.3 节）。`Yuliang/DART` 在 HuggingFace 上只提供含完整 DARTset 的大压缩包，无法单独下载 `texture&accessories/` 子目录，因此已预先上传到 R2。

```bash
mkdir -p "$FOREHOI_WORK_ROOT/DART"
rclone copy r2:pi3-forehoi-deps/dart-texture-accessories \
  "$FOREHOI_WORK_ROOT/DART/texture&accessories" \
  --progress --transfers 4
```

### 4.5 Blender（2.3G）

来源：Blender 官网。需要与当前版本一致，当前使用的是 **Blender 4.x**（从目录名 `blender-runtime-5.1` 和 `blender-bin` 判断）。

```bash
# 查看当前版本
ls "$FOREHOI_WORK_ROOT/blender-bin/"

# 从 Blender 官网下载对应 Linux x64 版本
# https://www.blender.org/download/
# 选择 Linux > blender-<version>-linux-x64.tar.xz
wget "https://download.blender.org/release/Blender4.x/blender-4.x.x-linux-x64.tar.xz" \
  -O /tmp/blender.tar.xz
tar -xf /tmp/blender.tar.xz -C "$FOREHOI_WORK_ROOT/blender-bin/"
```

### 4.6 BlenderProc（167M）

来源：GitHub `DLR-RM/BlenderProc`

```bash
cd "$FOREHOI_WORK_ROOT"
git clone https://github.com/DLR-RM/BlenderProc.git BlenderProc
cd BlenderProc
pip install -e .
```

### 4.7 manopth（257M）

来源：GitHub `hassony2/manopth`

```bash
cd "$FOREHOI_WORK_ROOT"
git clone https://github.com/hassony2/manopth.git manopth
cd manopth
pip install -e .
```

---

## 5. 新服务器完整还原步骤

### 5.1 拉取代码

```bash
git clone https://github.com/xiaofeng218/Pi3.git
cd Pi3
git checkout training
```

### 5.2 创建 conda 环境

```bash
conda env create -f conda/pi3-environment.yml
conda activate pi3
pip install -e dex-ycb-toolkit
```

### 5.3 设置路径变量

```bash
# 按需修改这两个根路径
export PI3_DATA_ROOT=/data/pi3-assets      # 模型和DexYCB数据存放位置
export FOREHOI_DATA_ROOT=/data/forehoi     # ForeHOI 生成数据存放位置

source asset_registry/env.sh

# ForeHOI 路径单独覆盖（生成数据和原始数据分开存放）
export FOREHOI_ROOT="$FOREHOI_DATA_ROOT/pi3x_raw"
export FOREHOI_OBJECT_MULTIVIEW_ROOT="$FOREHOI_DATA_ROOT/object_multiview_pyrender"
```

### 5.4 从 R2 下载模型权重

```bash
source asset_registry/env.sh

rclone copy r2:pi3-models/hamer "$HAMER_CACHE_DIR" --progress --transfers 4
rclone copy r2:pi3-models/pi3x  "$PI3X_CKPT"       --progress --transfers 4
```

### 5.5 从 R2 下载 ForeHOI 生成数据

```bash
mkdir -p "$FOREHOI_DATA_ROOT"

# 核心数据（训练必需）
rclone copy r2:pi3-forehoi/render                   "$FOREHOI_DATA_ROOT/render"                   --progress --transfers 8
rclone copy r2:pi3-forehoi/pi3x_raw                 "$FOREHOI_DATA_ROOT/pi3x_raw"                 --progress --transfers 4
rclone copy r2:pi3-forehoi/object_multiview_pyrender "$FOREHOI_DATA_ROOT/object_multiview_pyrender" --progress --transfers 8

# 可选（重新运行 pipeline 时需要）
rclone copy r2:pi3-forehoi/object_parts             "$FOREHOI_DATA_ROOT/object_parts"             --progress --transfers 4
rclone copy r2:pi3-forehoi/cache                    "$FOREHOI_DATA_ROOT/cache"                    --progress --transfers 4
```

### 5.6 重建 pi3x_raw 软链接

`pi3x_raw/` 下各序列的 `rgb/`、`depth/`、`mask/` 目录是指向 `render/` 的软链接，上传时被跳过，需要在新服务器重建：

```bash
bash asset_registry/dataset/rebuild_forehoi_symlinks.sh "$FOREHOI_DATA_ROOT"
```

脚本内容见 `asset_registry/dataset/rebuild_forehoi_symlinks.sh`。

### 5.7 更新 manifest.json 路径

`manifest.json` 中包含旧服务器的绝对路径，需要替换为新路径：

```bash
OLD_PREFIX="/mnt/2/data/dataset/ForeHOI/outputs/pi3x-pipeline"
NEW_PREFIX="$FOREHOI_DATA_ROOT"

sed -i "s|$OLD_PREFIX|$NEW_PREFIX|g" "$FOREHOI_DATA_ROOT/pi3x_raw/manifest.json"
```

### 5.8 下载 DexYCB

```bash
source asset_registry/env.sh
bash asset_registry/dataset/download_datasets.sh
```

### 5.9 验证

```bash
source asset_registry/env.sh
export FOREHOI_ROOT="$FOREHOI_DATA_ROOT/pi3x_raw"
export FOREHOI_OBJECT_MULTIVIEW_ROOT="$FOREHOI_DATA_ROOT/object_multiview_pyrender"

# 检查关键路径
test -f "$FOREHOI_ROOT/split.json"         && echo "split.json OK"
test -d "$FOREHOI_OBJECT_MULTIVIEW_ROOT"   && echo "object_multiview_pyrender OK"
test -f "$HAMER_ENCODER_CKPT"              && echo "HaMeR ckpt OK"
test -d "$DEXYCB_ROOT"                     && echo "DexYCB OK"

# import 检查
python -c "import torch; import pi3; import dex_ycb_toolkit; print('imports ok', torch.__version__)"

# ForeHOI 数据加载 smoke test
python - <<'PY'
import os
from pathlib import Path
from datasets.forehoi_dataset import ForeHOIDataset

forehoi_root = os.environ["FOREHOI_ROOT"]
ds = ForeHOIDataset(data_root=forehoi_root, frame_num=4, mode="train")
print(f"ForeHOI sequences: {len(ds)}")
sample = ds[0]
print(f"sample keys: {[v.keys() for v in sample[:1]]}")
PY
```

---

## 6. 上传命令汇总（一键脚本）

保存以下内容到 `asset_registry/upload_to_r2.sh`，在当前服务器执行：

```bash
#!/usr/bin/env bash
set -euo pipefail

FOREHOI_SRC="${FOREHOI_SRC:-/mnt/2/data/dataset/ForeHOI/outputs/pi3x-pipeline}"
DART_SRC="${DART_SRC:-/mnt/2/data/dataset/ForeHOI/DART}"
MODEL_SRC="${MODEL_SRC:-/mnt/2/projects/Pi3/data/model}"
R2="${R2_REMOTE:-r2}"

echo "==> Uploading model weights..."
rclone copy "$MODEL_SRC/hamer/_DATA"  "$R2:pi3-models/hamer"  --progress --transfers 4
rclone copy "$MODEL_SRC/pi3x"         "$R2:pi3-models/pi3x"   --progress --transfers 4

echo "==> Uploading ForeHOI render (35G, may take a while)..."
rclone copy "$FOREHOI_SRC/render"                    "$R2:pi3-forehoi/render"                    --progress --transfers 8 --multi-thread-streams 4

echo "==> Uploading ForeHOI pi3x_raw (meta only)..."
rclone copy "$FOREHOI_SRC/pi3x_raw"                  "$R2:pi3-forehoi/pi3x_raw"                  --skip-links --progress --transfers 4

echo "==> Uploading object_multiview_pyrender..."
rclone copy "$FOREHOI_SRC/object_multiview_pyrender" "$R2:pi3-forehoi/object_multiview_pyrender" --progress --transfers 8

echo "==> Uploading object_parts..."
rclone copy "$FOREHOI_SRC/object_parts"              "$R2:pi3-forehoi/object_parts"              --progress --transfers 4

echo "==> Uploading DART texture&accessories (1.9G)..."
rclone copy "$DART_SRC/texture&accessories"          "$R2:pi3-forehoi-deps/dart-texture-accessories" --progress --transfers 4

echo "==> Upload complete."
echo "Verify with: rclone size $R2:pi3-forehoi/ && rclone size $R2:pi3-forehoi-deps/"
```

---

## 7. 常见问题

### rclone 传输中断

重复执行同一 `rclone copy` 命令即可，rclone 会自动跳过已上传的文件（按 size + mtime 比对）。

### 训练时找不到 ForeHOI 图像

检查 `pi3x_raw/<shard>/rgb/` 下是否有软链接，且链接目标是否存在：

```bash
ls -la "$FOREHOI_ROOT/$(ls $FOREHOI_ROOT | grep large | head -1)/rgb/" | head -3
```

如无软链接，重新执行：

```bash
bash asset_registry/dataset/rebuild_forehoi_symlinks.sh "$FOREHOI_DATA_ROOT"
```

### manifest.json 路径不对

重新执行 5.7 节的 `sed` 命令，检查替换结果：

```bash
python3 -c "
import json
d = json.load(open('$FOREHOI_ROOT/manifest.json'))
print(d['sequences'][0]['render_dir'])
"
```

输出路径应指向当前服务器上的真实目录。

### 3D-FRONT 申请未收到回复

3D-FRONT 下载 URL 由 Alibaba Tianchi 手动发放，通常需要 1-2 周。可在申请邮件中抄送一下 PI 信息并说明用途（学术研究）以加快审核。当前 `front-3d/download.sh` 中的 URL 已过期，收到新 URL 后需替换。

---

## 6. ForeHOI 预处理代码仓库

ForeHOI 目录下的预处理代码（`pi3x-preproc/`、`preprocess/`、`debug-preproc/` 等）需要单独的 git 仓库管理，与数据目录分开。

### 6.1 仓库内容

| 目录 / 文件 | 说明 |
|-------------|------|
| `pi3x-preproc/` | 核心 pipeline：ForeHOI → Pi3X 格式转换 |
| `preprocess/` | ForeHOI shard 重建、hand/object 解析工具 |
| `debug-preproc/` | 在 Front3D 场景中渲染并验证输出 |
| `download_forehoi_minimal.py` / `.sh` | graspxl_renders + Objaverse 下载脚本 |
| `download_objaverse_object.py` | 单独下载 Objaverse asset |
| `example_loader.py` | ForeHOI shard 读取示例 |
| `run.sh`, `view_rerun_web.sh` | 常用运行脚本 |
| `front-3d/download.sh`, `front-3d/3D-FRONT-readme.md` | 3D-FRONT 申请/下载说明 |
| `docs/` | 说明文档 |
| `README.md` | 仓库说明 |

**不追踪（.gitignore）：** `outputs/`、`graspxl_renders/`、`front-3d/3D-FRONT*/`、`front-3d/3D-FUTURE*/`、`DART/`、`objaverse_assets/`、`blender-bin/`、`blender-runtime-5.1/`、`samples/`、`BlenderProc/`、`manopth/`、`template mesh/`

### 6.2 本地初始化

在当前服务器执行（已包含 `.git` 空目录，先清理再初始化）：

```bash
cd /mnt/2/data/dataset/ForeHOI

# 清理空的 .git 目录（不是真正的 git repo）
rm -rf .git

# 初始化
git init
git add .gitignore README.md run.sh view_rerun_web.sh
git add download_forehoi_minimal.py download_forehoi_minimal.sh download_objaverse_object.py example_loader.py
git add pi3x-preproc/ preprocess/ debug-preproc/ docs/
git add front-3d/download.sh front-3d/3D-FRONT-readme.md
git commit -m "init: ForeHOI pi3x preprocessing pipeline"
```

### 6.3 创建 GitHub 远程仓库并推送

在 GitHub 上创建新的私有仓库（建议名称：`forehoi-pi3x-preproc`），然后：

```bash
cd /mnt/2/data/dataset/ForeHOI

git remote add origin https://github.com/xiaofeng218/forehoi-pi3x-preproc.git
git branch -M main
git push -u origin main
```

### 6.4 新服务器克隆

```bash
git clone https://github.com/xiaofeng218/forehoi-pi3x-preproc.git
cd forehoi-pi3x-preproc

# 安装依赖（需要在 pi3 conda 环境中）
conda activate pi3
pip install blenderproc rerun-sdk objaverse
```

然后按第 4 节下载各原始依赖数据到同一目录下。
