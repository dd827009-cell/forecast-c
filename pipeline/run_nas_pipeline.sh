#!/usr/bin/env bash
# 完整資料取得流程：FTP 只下 .pdb 篩 cohort → 只下 cohort 完整 → 轉 h5。
#
# 密碼安全：放 ~/.netrc（不在本檔、不進 git）：
#   machine cad.csie.ntu.edu.tw login d13945010 password 你的密碼
#   然後 chmod 600 ~/.netrc
# lftp 會自動讀取，本腳本完全不碰明碼。
#
# 用法：改下面「設定」5 行 → 在 forecast-c 根、轉檔 venv 啟用下執行：
#   bash pipeline/run_nas_pipeline.sh
set -euo pipefail

# ===================== 設定（只改這裡）=====================
FTP_HOST="cad.csie.ntu.edu.tw"
FTP_USER="d13945010"
# 從 lftp `pwd` 貼上實際路徑（中文資料夾用實際名，不要 %A9%FA 那種編碼）：
REMOTE_BASE="/eye2/請貼實際資料夾名/CGMHOCT_Heideberg/Patients-1"
EXCEL="/path/to/EYLEA 8mg 恩慈整理完成 (3).xlsx"   # 治療 Excel（含病歷號欄）
SSD="/ssd/octdata"                                  # 本機 SSD 工作目錄（要有幾百 GB 空間）
WORKERS=16
# ==========================================================

REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$SSD"/{pdb_only,cohort_raw,h5_output}

echo "===== [1/6] 只下載全部 .pdb（~3GB，保留資料夾結構）====="
lftp -u "$FTP_USER" "$FTP_HOST" -e "set net:connection-limit 4; mirror -I '*.pdb' --parallel=4 --continue '$REMOTE_BASE' '$SSD/pdb_only'; bye"

echo "===== [2/6] 掃 .pdb → 病歷號 index.csv ====="
python "$REPO/pipeline/scan_pdb.py" --input "$SSD/pdb_only" --repo-root "$REPO" --workers "$WORKERS" --out "$SSD/index.csv"

echo "===== [3/6] 治療 Excel → cohort 病歷號清單 ====="
python "$REPO/pipeline/cohort_list.py" --excel "$EXCEL" --out "$SSD/cohort.txt"

echo "===== [4/6] 篩出 cohort 的 .pat 清單 ====="
python "$REPO/pipeline/filter_pats.py" --index "$SSD/index.csv" --cohort "$SSD/cohort.txt" --out "$SSD/cohort_pats.txt"

echo "===== [5/6] 只下載 cohort 完整 .pat（含大 .sdb，可續跑）====="
bash "$REPO/pipeline/download_cohort.sh" \
    "$SSD/cohort_pats.txt" "$REMOTE_BASE" "$SSD/cohort_raw" "$FTP_HOST" "$FTP_USER" "$SSD/pdb_only"

echo "===== [6/6] 轉檔 cohort → h5（平行+冪等可續跑）====="
( cd "$REPO/pdb_to_h5" && python -m heyex_pipeline \
    --input "$SSD/cohort_raw" --output "$SSD/h5_output" \
    --workers "$WORKERS" --manifest-checkpoint-interval 10000 --verify-samples 20 )

echo ""
echo "✅ 完成。cohort h5 在：$SSD/h5_output"
echo "下一步（普查）："
echo "  python -m forecast_c.census.a1_census --h5-dir $SSD/h5_output --out census_out"
echo "（轉完確認無誤後，可刪 $SSD/cohort_raw 與 $SSD/pdb_only 省空間）"
