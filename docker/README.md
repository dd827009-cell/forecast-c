# 本機開發/驗證環境 (Docker + GPU)

3060 Ti 8GB + Windows。用 Docker 跑一個含 Python+torch+CUDA 的容器，與 L40/Linux 同環境，
躲掉 Windows 上 torch×cv2×h5py 的 DLL segfault。

> ⚠️ 此環境能做：dataloader self-test、backbone batch=1 forward、stage0 pilot 跑、模塊 C/D/F 開發。
> 做不到：Phase 1 正式訓練 backbone（ViT-L 雙分支 60 frame，8GB 不夠 → 那要 L40）。

---

## 一次性安裝（需系統管理員 + 重開機，在 Windows 上做一次）

### 1. 裝 WSL2（Docker GPU 的底層）
以**系統管理員** PowerShell：
```powershell
wsl --install
```
裝完**重開機**。（Docker Desktop 的 GPU passthrough 走 WSL2，這步不能跳。）

### 2. 裝 Docker Desktop
下載 <https://www.docker.com/products/docker-desktop/> 安裝，啟動後：
- Settings → General → 勾 **Use the WSL 2 based engine**
- Settings → Resources → GPU 相關選項開啟（新版預設支援 NVIDIA）

### 3. 確認 GPU 能進容器
```powershell
docker run --rm --gpus all pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
看到 `True NVIDIA GeForce RTX 3060 Ti` 就成功。

---

## 建 image（每次改 requirements 才要重建）

在 repo 根目錄（`C:\Users\Administrator\Desktop\pretrain`）的 PowerShell：
```powershell
docker build -f docker/Dockerfile -t octcube-dev docker/
```
> build context 指到 `docker/`，只送 requirements 進去，不會把 1.3G pilot h5 / 4G checkpoint 塞進 build。

---

## 進容器幹活

```powershell
docker run --gpus all -it --rm -v ${PWD}:/workspace octcube-dev bash
```
- `-v ${PWD}:/workspace`：把整個 repo 掛進容器 `/workspace`，存檔即時同步，容器刪了程式碼還在。
- `--rm`：離開即清掉容器（image 留著）。

進去後先驗 3 件事：
```bash
# 0) GPU 看得到嗎
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 1) dataloader self-test (零資料零 GPU, 最快打勾)
python stage1/shard_dataset.py --self-test

# 2) 載 OCTCube 權重 + pos-embed 插值 + batch=1 forward
#    (路徑自動從 repo 根推導; 權重需在 OCTCubeM-main/OCTCube.pth)
python stage1/smoke_test_load.py
```

---

## 用 VS Code 操作（推薦，免打 docker 指令）

前置一樣要先裝好 **WSL2 + Docker Desktop**（上面步驟 1、2），然後：
1. VS Code 裝 **Dev Containers** 擴充（`ms-vscode-remote.remote-containers`）。
2. 開啟本專案資料夾 → 左下角綠色按鈕（或 F1）選 **Reopen in Container**。
3. VS Code 會自動用 `.devcontainer/devcontainer.json`（重用 `docker/Dockerfile`）build + 進容器，
   編輯器/終端機/Python 直譯器全接進容器，GPU 也已 `--gpus all` 掛好。

進去後在 VS Code 內建終端機跑驗證指令（同下方「進容器幹活」）。
> VS Code 只是 Docker 的前端，**不取代** Docker 引擎；WSL2 + Docker Desktop 仍是必裝。

## 備忘
- 改 torch/CUDA 版本：編輯 `docker/Dockerfile` 的 `FROM` tag。
- 加套件：改 `docker/requirements-docker.txt` 後重 build。
- `flash-attn` 故意不裝（smoke test 已 stub）；真要在 L40 跑官方 flash 路徑再另裝。
- 想固定版本給 L40 重現：在容器內 `pip freeze > docker/requirements.lock.txt`。
