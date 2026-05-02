#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-${DEXYCB_ROOT:-data/dataset/dexycb}}"

# 检查并安装/更新 gdown
if ! command -v gdown &> /dev/null
then
    python -m pip install -U gdown
fi

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

# 文件 ID 列表
FILE_IDS=(
# "1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk"
# "1CPqLjsaYNjE3xSJbuWmqaMsGvyGIxiKL"
# "1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ"
# "1cAzlQBpcTatI5ykYQ8ziQiHLUG_a_UpM"
"1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62"
"1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq"
"14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA"
"1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS"
)

echo "开始下载（支持断点续传模式）..."

for ID in "${FILE_IDS[@]}"
do
    echo "正在下载: $ID"
    # 直接传入 ID，不加 --id
    # --continue 用于断点续传
    gdown "$ID" --continue
done

echo "任务完成。"
