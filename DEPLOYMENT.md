# VITS2 語音合成專案 (VITS2 TTS Project)

## 1. 專案簡介

本專案基於 **VITS2** 框架實現端對端語音合成系統。專案採用 Docker 容器化方案，確保訓練與推論環境的一致性，並簡化了 GPU 驅動與依賴項的配置流程。

## 2. 環境需求

在使用 Docker 部署前，請確保您的主機符合以下條件：

* **作業系統**: Linux (推薦 Ubuntu 22.04+) 或 Windows (需開啟 WSL2)。
* **硬體**: NVIDIA GPU (顯存建議 12GB 以上，依 Batch Size 而定)。
* **軟體**:
  * [Docker Desktop](https://www.docker.com/products/docker-desktop/) 或 Docker Engine。
  * [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (用於讓 Docker 容器調用 GPU 資源)。

## 3. 部署流程

### 3.1 構建鏡像

在專案根目錄下，執行 Docker Compose 構建專案鏡像。由於本專案使用了 **Profiles**，建議在構建時指定對應的 Profile：

* **構建訓練環境 (vits2-train)**:

    ```powershell
    docker compose --profile train build
    ```

* **構建推論環境 (vits2-inference)**:

    ```powershell
    docker compose --profile inference build
    ```

* **一次性構建所有服務**:

    ```powershell
    docker compose --profile "*" build
    ```

此步驟會自動安裝 Python 3.11 環境、CUDA 支援庫、`espeak-ng` (語音處理)、`ffmpeg` 以及專案所需的 Python 套件。

### 3.2 數據預處理 (可選)

#### 音訊重採樣

VITS2 訓練通常需要統一的採樣率 (例如 22050Hz)。使用內建的重採樣服務：

```powershell
docker compose --profile resample up
```

* **預設路徑**: `datasets/ljs_base/wavs` → `datasets/ljs_base/wavs_22k` (22050Hz)。
* 若需修改路徑，請編輯 `docker-compose.yml` 中的 `vits2-resample` 服務指令。

#### 數據集標註生成

若有特定數據集 (如 Matsu 數據集)，可執行專用的預處理腳本：

```powershell
docker compose run --rm vits2-train python scripts/prepare_matsu.py
```

### 3.3 啟動訓練 (Training)

啟動訓練容器：

```powershell
docker compose --profile train up
```

* **配置檔**: 預設使用 `datasets/ljs_base/config.yaml`。
* **日誌與權重**: 訓練過程中的 Tensorboard 日誌與模型權重會儲存於 `logs/ljs_base/` 目錄下。
* **中斷恢復**: 再次執行指令會自動從最新的 Checkpoint 恢復訓練。

### 3.4 啟動推論環境 (Inference / Jupyter Lab)

啟動 Jupyter Lab 以進行互動式推論測試：

```powershell
docker compose --profile inference up
```

* 啟動後訪問 `http://localhost:8888`。
* 可開啟 `inference.ipynb` 進行模型效果評估。

## 4. 訓練語言與文本處理 (Language & Text Processing)

本專案支援多語言訓練，主要透過 `config.yaml` 中的 `data` 欄位進行配置：

### 4.1 調整語種

在 `config.yaml` 中修改 `language` 參數。此參數需符合 `espeak-ng` 支援的代碼：

* **英文**: `en-us`
* **中文 (普通話)**: `cmn`
* **日文**: `ja`
* **馬祖話 (閩東語)**: `nan` (或使用專用的 `matsu_cleaner`)

### 4.2 配置文本清洗器 (Text Cleaners)

`text_cleaners` 定義了文本轉音標的流程。常見配置如下：

* **一般語種 (使用 espeak)**:

    ```yaml
    text_cleaners:
      - phonemize_text # 調用 phonemizer 轉音標
      - add_spaces     # 在符號間增加空格
      - tokenize_text  # 轉為 ID 序列
    ```

* **自定義語種 (如馬祖話)**:
    若不需要音標轉換，可改用 `matsu_cleaner` 進行字元層級處理：

    ```yaml
    text_cleaners:
      - matsu_cleaner
      - tokenize_text
    ```

## 5. 專案結構說明

* `train.py`: 單人說話者訓練入口。
* `train_ms.py`: 多人說話者訓練入口。
* `data_utils.py`: 數據載入與資料增強邏輯。
* `text/`: 文本清理、注音/音標轉換模組。
* `model/`: VITS2 模型架構實作 (Flow-based, SDP, etc.)。
* `scripts/`: 資料重採樣與特定數據集生成工具。
* `docker-compose.yml`: 定義訓練、推論與工具服務。

## 6. 常見問題與調整

* **共享記憶體 (Shared Memory)**:
  * `shm_size` 在 `docker-compose.yml` 中預設為 `8gb`。若在 DataLoader 讀取數據時發生 `Bus Error`，請嘗試調大此數值或檢查主機記憶體。
* **GPU 配置**:
  * 目前預設使用 `NVIDIA_VISIBLE_DEVICES=0`。
  * 若主機有多張顯卡，可調整 `device_ids` 或環境變數來選擇指定 GPU。
* **路徑掛載**:
  * 本機目錄已與容器內的 `/app` 同步，修改本機代碼或數據會即時反映在容器中。

---
*Last updated: 2026-04-01*
