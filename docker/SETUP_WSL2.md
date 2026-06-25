# 本機環境設定:WSL2 + Docker Engine（免 Docker Desktop）

目標:在 WSL2 裡裝 Docker Engine + NVIDIA Container Toolkit，讓 3060 Ti 能在容器內用 CUDA。
照順序做，每階段都有「驗證」確認過了再往下。

圖示:🪟 = Windows PowerShell ／ 🐧 = WSL Ubuntu 終端機

---

## 階段 0:前置（已具備，僅確認）
- NVIDIA 驅動 610.47 已裝 ✅（GPU 透過它分享給 WSL2，**WSL 內不要再裝 Linux 驅動**）

---

## 階段 1:裝 WSL2 🪟（系統管理員 PowerShell）
開始功能表搜尋 PowerShell → 右鍵「以系統管理員身分執行」：
```powershell
wsl --install
```
→ **重開機**。重開後 Ubuntu 視窗會自動跳出，要你設一組 **Linux 帳號 + 密碼**（隨意設、記住，之後 sudo 要用）。

**驗證** 🪟：
```powershell
wsl -l -v        # 應看到 Ubuntu, VERSION = 2
```

---

## 階段 2:WSL 內裝 Docker Engine 🐧
開「Ubuntu」（開始功能表）或在 PowerShell 打 `wsl` 進入，然後：
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y curl ca-certificates

# 官方一鍵安裝腳本
curl -fsSL https://get.docker.com | sh

# 讓你的帳號免 sudo 用 docker
sudo usermod -aG docker $USER
```
群組要重登才生效 —— 回 🪟 PowerShell 打 `wsl --shutdown`，再重開 Ubuntu 終端機。

**啟動 daemon + 驗證** 🐧：
```bash
sudo service docker start
docker run --rm hello-world      # 看到 "Hello from Docker!" 即成功
```

---

## 階段 3:裝 NVIDIA Container Toolkit 🐧（讓 --gpus 生效）
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
```

**驗證 GPU 進得了容器** 🐧：
```bash
docker run --rm --gpus all pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
看到 `True NVIDIA GeForce RTX 3060 Ti` → 環境完成！

---

## 階段 4:build 專案 image + 跑驗證 🐧
進到 repo（在 WSL 裡 C 槽掛在 /mnt/c）：
```bash
cd /mnt/c/Users/Administrator/Desktop/pretrain

# build image（context 指 docker/，不會把 1.3G 資料塞進去）
docker build -f docker/Dockerfile -t octcube-dev docker/

# ① dataloader self-test（零資料零 GPU，最快打勾）
docker run --gpus all --rm -v "$(pwd)":/workspace octcube-dev \
  python stage1/shard_dataset.py --self-test

# ② 載 OCTCube 權重 + pos-embed 插值 + batch=1 forward
docker run --gpus all --rm -v "$(pwd)":/workspace octcube-dev \
  python stage1/smoke_test_load.py

# 互動進容器（之後開發模塊用）
docker run --gpus all -it --rm -v "$(pwd)":/workspace octcube-dev bash
```

> ⚠️ /mnt/c 掛載比 WSL 原生檔系統慢，做小規模驗證沒差；之後若要大量讀寫再考慮搬到 WSL home。

---

## 選用:免每次手動啟動 docker（開 systemd）🐧
預設每次開 WSL 要 `sudo service docker start`。要自動啟動，編輯 `/etc/wsl.conf`：
```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true
EOF
```
然後 🪟 `wsl --shutdown` 重啟 WSL。之後 docker 會隨 WSL 自動起，且 `systemctl` 可用。

---

## 常見雷
- **WSL 內裝 Linux NVIDIA 驅動** → 會搞壞 GPU 透傳。只裝 `nvidia-container-toolkit`。
- **`docker` 找不到 / Cannot connect to daemon** → 忘了 `sudo service docker start`（或沒開 systemd）。
- **`docker` 要 sudo** → `usermod -aG docker` 後沒重登；`wsl --shutdown` 再進來。
- **build 很慢/context 很大** → 確認指令是 `-f docker/Dockerfile ... docker/`（context 是 docker/，不是 repo 根）。
</content>
