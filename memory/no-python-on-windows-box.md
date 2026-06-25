---
name: no-python-on-windows-box
description: "Native Windows has no Python, but WSL2+Docker octcube-dev now runs Python/torch/CUDA locally"
metadata: 
  node_type: memory
  type: project
  originSessionId: 558a1d33-56be-42c4-8cf9-9766819f4848
---

這台 Windows 11 開發機的**原生環境沒有可用 Python**（只有 Microsoft Store stub；無 conda/venv/註冊表）。

**但（2026-06 更新）已建好 WSL2 + Docker Engine + `octcube-dev` 容器** → 本機現在**可以**跑 Python/torch/CUDA。
**✅ 已驗證可用指令（2026-06-16；torch 2.4.0；映像 `octcube-dev:latest`；CPU dummy 測不需 --gpus）**：
```
wsl -e bash -lc "docker run --rm -v /mnt/c/Users/Administrator/Desktop/pretrain:/work -w /work octcube-dev:latest python -m latent_dynamics.tests.run_tests"
```
- ⚠️ **掛載用絕對路徑 `-v /mnt/c/...:/work`，別用 `$(pwd)`**（Bash 工具 cwd 會殘留別目錄 → 掛錯 repo）。
- Windows 主機的 `python3` 是空殼、且原生 torch×cv2/h5py DLL segfault → **一律走容器**。預設 WSL 的 python3 也沒 torch，**只有容器有**。
- GPU（RTX 3060 Ti 8GB）在容器內 `torch.cuda.is_available()=True`。
- stage0 M1–M7、dataloader self-test、各模塊單元測、train harness smoke **都已能在容器內實跑**（pilot/dummy 級）。
- 容器是 docker-in-WSL（**非** Docker Desktop）；`docker` 指令在 WSL 裡，主機端透過 `wsl.exe docker ...` 呼叫。

**仍做不到的（要 L40）**：在 3060 Ti **訓練 backbone**（ViT-L fwd+bwd 連 batch1 都需 ~20GB；GPU 探測工具 `stage1/gpu_batch_probe.py`）。
torch×cv2/h5py 在**原生 Windows** 會 DLL segfault；**Linux 容器內無此問題**，故一律走容器。
⚠️ WSL2 `/mnt/c` 對「容器內新建資料夾」寫回不可靠 → 要持久化的輸出，**先從主機端建好目錄**再讓容器寫，或寫到容器內 `/tmp`。
相關：[[authoritative-roadmap]]、[[repo-build-state]]。
