# 新電腦一鍵環境安裝（Windows，什麼都沒裝）

目標：讓一台全新的 Windows 電腦能跑 `pipeline/run_all_paths.sh`（NAS 取檔 + 篩 cohort + pat_log）。
路線＝**WSL2 + Ubuntu + lftp + python3**。中間需要**重開機一次**（WSL 安裝天性）。

## 步驟
### 1. 裝 WSL2 + Ubuntu（Windows 端，系統管理員）
先把專案抓到這台（見下方「拿到專案」），然後用**系統管理員** PowerShell：
```powershell
powershell -ExecutionPolicy Bypass -File pipeline\setup\install_windows.ps1
```
→ 完成後**重新開機**。

### 2. 首次啟動 Ubuntu
重開後從「開始」開 **Ubuntu**，第一次會請你設一組 Linux 使用者名稱/密碼（記住 Linux 密碼，`sudo` 要用）。

### 3. 裝工具 + 設 NAS 密碼（Ubuntu 端）
```bash
bash forecast-c/pipeline/setup/setup_wsl.sh            # 取檔/掃描/log 所需（lftp + python3）
# 若這台也要轉 h5：
bash forecast-c/pipeline/setup/setup_wsl.sh --with-h5  # 另裝 numpy/h5py/pandas... 較大較久
```
它會裝好 `lftp`/`python3`、互動式建立 `~/.netrc`（輸入 NAS 密碼、自動 chmod 600），最後測一次連線。

### 4. 開跑
```bash
cd forecast-c
# 編輯 pipeline/run_all_paths.sh 最上面：EXCELS 路徑、SSD 工作目錄、DO_H5=0
bash pipeline/run_all_paths.sh 2>&1 | tee /mnt/d/octrun.log
```

## 拿到專案（兩選一）
- **git clone**（需先把 repo push 到 GitHub）：
  ```bash
  git clone https://github.com/dd827009-cell/forecast-c.git
  ```
- **直接複製資料夾**：把整個 `forecast-c` 資料夾拷到新電腦即可。

> ⚠️ **兩份 Excel 不在 git 裡**（治療 metadata，含真實姓名）→ 一定要**另外手動複製**到新電腦，
> 放桌面即可（WSL 路徑 `/mnt/c/Users/你的帳號/Desktop/...`），再填進 `run_all_paths.sh` 的 `EXCELS`。

## 只想要「取檔+pat_log」不碰訓練
不需要 Docker、不需要 OCTCube/教師權重、不需 `--with-h5`。上面 1–4 步 + `DO_H5=0` 就夠了。
訓練環境（`octcube-dev` Docker、權重）是另一條線，見 repo 根 `CLAUDE.md §5/§6`。
