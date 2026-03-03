# 使用具備 CUDA 支援的 NVIDIA 基礎鏡像
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 設置環境變量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    git \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 設置 Python 3.11 為預設 python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# 設置工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝依賴
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# 複製專案文件
COPY . .

# 預編譯 Numba 模組 (可選，加快啟動速度)
# RUN python monotonic_align.py

# 預設執行指令 (可根據需求修改)
CMD ["python", "train.py", "-c", "datasets/ljs_base/config.yaml", "-m", "ljs_base"]
