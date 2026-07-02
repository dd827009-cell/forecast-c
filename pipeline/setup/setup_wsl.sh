#!/usr/bin/env bash
# setup_wsl.sh — 在 Ubuntu(WSL) 裡裝好本 pipeline 需要的一切，並建立 ~/.netrc。
# 用法（在 Ubuntu 裡）:
#   bash forecast-c/pipeline/setup/setup_wsl.sh            # 只裝取檔/掃描/log 所需（lftp + python3）
#   bash forecast-c/pipeline/setup/setup_wsl.sh --with-h5  # 另外裝轉 h5 的 Python 套件（較大/較久）
#
# 取檔+掃描+cohort+pat_log 只需 lftp + python3 標準庫（scan_pdb/cohort_list/pat_log 都純標準庫）。
# 轉 h5 才需要 numpy/h5py/pandas... → 用 --with-h5。

set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

echo "===== [1/4] 安裝系統工具（lftp / python3 / git）====="
sudo apt-get update
sudo apt-get install -y lftp python3 python3-pip git coreutils gawk sed grep

if [ "${1:-}" = "--with-h5" ]; then
  echo "===== [1b] 安裝轉 h5 的 Python 套件（--with-h5）====="
  # 失敗不致命：取檔/掃描不需要這些，之後可再補
  pip3 install --user --upgrade pip
  pip3 install --user numpy h5py pandas simplejson iopath eyepy \
    || echo "  ⚠️ 部分套件安裝失敗（eyepy 常見）；取檔/掃描不受影響，轉 h5 前再處理。"
fi

echo ""
echo "===== [2/4] 建立 ~/.netrc（NAS 密碼，chmod 600、不進 git）====="
if [ -f "$HOME/.netrc" ] && grep -q "cad.csie.ntu.edu.tw" "$HOME/.netrc"; then
  echo "  ✓ ~/.netrc 已有 cad.csie.ntu.edu.tw，略過。"
else
  read -rp "  NAS 帳號 [預設 d13945010]: " NUSER
  NUSER="${NUSER:-d13945010}"
  read -rsp "  NAS 密碼（不會顯示）: " NPASS; echo
  # 保留既有其他 machine 行，只補這台
  touch "$HOME/.netrc"
  printf 'machine cad.csie.ntu.edu.tw login %s password %s\n' "$NUSER" "$NPASS" >> "$HOME/.netrc"
  chmod 600 "$HOME/.netrc"
  echo "  ✓ 已寫入 ~/.netrc 並設 600。"
fi

echo ""
echo "===== [3/4] 驗證 ====="
echo "  lftp   : $(command -v lftp || echo 缺)"
echo "  python3: $(python3 --version 2>&1)"
echo "  repo   : $REPO"

echo ""
echo "===== [4/4] 測連線（列一個小資料夾，40 秒逾時，不會卡死）====="
if timeout 40 lftp cad.csie.ntu.edu.tw \
     -e "set ssl:verify-certificate no; set net:timeout 15; set net:max-retries 0; set ftp:charset big5; set file:charset utf-8; cls '/eye2/eye4(cad5)/ike/HXD00002/'; bye" 2>/dev/null | head; then
  echo "  ✓ 連線 OK。"
else
  echo "  ⚠️ 沒印出檔名/逾時 → 檢查：① ~/.netrc 密碼對不對（ls -l ~/.netrc 要 -rw-------）；"
  echo "     ② 這台（內網）連不連得到 NAS： getent hosts cad.csie.ntu.edu.tw"
fi

echo ""
echo "✅ 環境就緒。接下來："
echo "  1) 把兩份 Excel 複製到這台（放 Windows 桌面即可，WSL 路徑 /mnt/c/Users/你/Desktop/...）。"
echo "  2) 編輯 $REPO/pipeline/run_all_paths.sh 最上面設定："
echo "     - EXCELS 兩個路徑、SSD 工作目錄（例 /mnt/d/octdata，要幾百 GB）、DO_H5 先設 0"
echo "  3) 開跑： cd $REPO && bash pipeline/run_all_paths.sh 2>&1 | tee /mnt/d/octrun.log"
