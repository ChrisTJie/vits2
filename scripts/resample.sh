#!/bin/bash

# 設定參數
IN_DIR=${1:-"/app/datasets"}
OUT_DIR=${2:-"/app/datasets_resampled"}
TARGET_SR=${3:-22050}
THREADS=$(nproc)

echo "開始重採樣任務..."
echo "輸入目錄: $IN_DIR"
echo "輸出目錄: $OUT_DIR"
echo "目標採樣率: $TARGET_SR"
echo "平行執行緒: $THREADS"

# 建立輸出目錄並複製目錄結構
mkdir -p "$OUT_DIR"
cd "$IN_DIR" && find . -type d -exec mkdir -p "$OUT_DIR/{}" \;

# 使用 ffmpeg 進行平行重採樣
# -ar: 音訊採樣率
# -ac 1: 強制單聲道 (通常 VITS 需要)
# -hide_banner -loglevel error: 減少輸出雜訊
find . -name "*.wav" -print0 | xargs -0 -I {} -P "$THREADS" \
    ffmpeg -i "{}" -ar "$TARGET_SR" -ac 1 -hide_banner -loglevel error -y "$OUT_DIR/{}"

echo "所有音訊處理完成！"
