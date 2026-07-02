#!/usr/bin/env bash
# 多路徑完整編排：對 nas_bases.txt 每個 base 做「先掃再取」，最後對兩份 Excel 篩 cohort 下載完整 + 轉 h5。
#
#   每個 base:  list_pats(cls 列 .pat) → 直接下所有 .pdb → scan_pdb(病歷號) → per-base 病歷號 log
#   全域:       Excel→cohort 病歷號 → 各 base 篩中 → 只下 cohort 完整(.sdb) → 轉 h5 → 合併總 log
#
# 密碼安全：放 ~/.netrc（不在本檔、不進 git）：
#   machine cad.csie.ntu.edu.tw login d13945010 password 你的密碼   → chmod 600 ~/.netrc
#
# 用法：改下面「設定」→ 在 forecast-c 根、轉檔 venv（有 pandas/eyepy）下執行：
#   bash pipeline/run_all_paths.sh
# 全程可續跑：列檔/下載/轉檔都冪等，中斷再跑即可。
set -uo pipefail

# ===================== 設定（只改這裡）=====================
FTP_HOST="cad.csie.ntu.edu.tw"
FTP_USER="d13945010"
BASES_FILE="pipeline/nas_bases.txt"                 # 要處理的遠端 base 清單
EXCELS=(                                            # 兩份治療 Excel（含病歷號/Chart no. 欄）
  "/mnt/c/Users/Administrator/Desktop/EYLEA  8mg 恩慈整理完成 (3).xlsx"
  "/mnt/c/Users/Administrator/Desktop/全體系case pooling (20250811更新) (2).xlsx"
)
SSD="/mnt/d/octdata"                                # 工作目錄（D 槽，771GB；WSL 用 /mnt/d 不是 D:\）
WORKERS=16                                          # scan_pdb / 轉檔 平行核數
PAR=2                                               # FTP 平行工作數（溫和；NAS 會鎖登入太頻繁的，別超過 3）
CHUNK_CHARSET="big5"                                # NAS 檔名編碼（CGMH 中文路徑）
DO_H5=0                                             # 1=最後轉 h5；0=只到下載完整為止
# 分階段（可用環境變數覆寫，例： PHASE=pdb bash pipeline/run_all_paths.sh）：
#   all    = 全跑（下 pdb+病歷號log → 對 Excel → 下完整）  ← 預設
#   pdb    = 只做前半：下所有 .pdb + 解病歷號 + 出病歷號 log，然後停（先檢查再決定）
#   match  = 讀已存在 index.csv → 對 Excel → 出「相符 .pat 清單」就停，**不下載任何完整 .pat**（你只要清單就用這個）
#   cohort = 讀已存在 index.csv → 對 Excel → 下載相符者的完整 .pat（含 .sdb 影像）
PHASE="${PHASE:-all}"
# ==========================================================

REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$SSD"
slug() { printf '%s' "$1" | sed 's#[^A-Za-z0-9._-]#_#g; s#__*#_#g; s#^_##; s#_$##'; }

# base 清單（忽略 # 與空行）
mapfile -t BASES < <(grep -vE '^\s*(#|$)' "$BASES_FILE")
echo "===== 將處理 ${#BASES[@]} 個遠端 base ====="
printf '  %s\n' "${BASES[@]}"

# ---------- 階段 1（PHASE=all/pdb）：每 base 列 .pat → 下所有 .pdb → 解病歷號 → per-base log ----------
if [ "$PHASE" = "all" ] || [ "$PHASE" = "pdb" ]; then
for BASE in "${BASES[@]}"; do
  SL="$(slug "$BASE")"; D="$SSD/$SL"; mkdir -p "$D/pdb_only"
  echo ""; echo "########## base: $BASE  (工作目錄 $D) ##########"

  echo "----- [1] 列出 .pat（cls，快，不遞迴）-----"
  bash "$REPO/pipeline/list_pats.sh" "$BASE" "$D/pat_list_rel.txt" \
      "$FTP_HOST" "$FTP_USER" "$CHUNK_CHARSET" || {
    echo "  ⚠️ 列 .pat 失敗，跳過此 base"; continue; }
  if [ ! -s "$D/pat_list_rel.txt" ]; then
    echo "  此 base 無直接 .pat，略過下載/掃描。"; continue
  fi

  echo "----- [2] 直接下載所有 .pdb（mirror -I '*.pdb'，平行、可續跑）-----"
  bash "$REPO/pipeline/mirror_list.sh" "$D/pat_list_rel.txt" "$BASE" "$D/pdb_only" \
      "$FTP_HOST" "$FTP_USER" '*.pdb' "$PAR" "$CHUNK_CHARSET"

  echo "----- [3] 掃 .pdb → 病歷號 index.csv -----"
  python3 "$REPO/pipeline/scan_pdb.py" --input "$D/pdb_only" --repo-root "$REPO" \
      --workers "$WORKERS" --out "$D/index.csv"

  echo "----- [4] per-base 病歷號 log -----"
  python3 "$REPO/pipeline/pat_log.py" --index "$SL=$D/index.csv" --out "$D/pat_log.csv"
done
fi

# 收集已存在的 index.csv（讓 PHASE=cohort 單獨跑也拿得到）
INDEX_ARGS=()
for BASE in "${BASES[@]}"; do
  SL="$(slug "$BASE")"; D="$SSD/$SL"
  [ -s "$D/index.csv" ] && INDEX_ARGS+=(--index "$SL=$D/index.csv")
done

# PHASE=pdb：只做前半 → 出「不含 matched 的病歷號總 log」後停。
if [ "$PHASE" = "pdb" ]; then
  echo ""; echo "===== [PHASE=pdb] 合併總病歷號 log（尚未對 Excel）====="
  python3 "$REPO/pipeline/pat_log.py" "${INDEX_ARGS[@]}" --out "$SSD/pat_log_all.csv"
  echo ""
  echo "✅ 前半完成。病歷號總表：$SSD/pat_log_all.csv"
  echo "   檢查沒問題後，跑後半： PHASE=cohort bash pipeline/run_all_paths.sh"
  exit 0
fi

# ---------- 階段 2：Excel → cohort 病歷號 ----------
echo ""; echo "===== [全域] 兩份 Excel → cohort 病歷號清單 ====="
python3 "$REPO/pipeline/cohort_list_standalone.py" "$SSD/cohort.txt" "${EXCELS[@]}"

# PHASE=match：只出「相符的 .pat 清單」就停，**完全不下載任何完整 .pat**。
if [ "$PHASE" = "match" ]; then
  echo ""; echo "===== [PHASE=match] 相符清單（不下載）====="
  python3 "$REPO/pipeline/pat_log.py" "${INDEX_ARGS[@]}" \
      --cohort "$SSD/cohort.txt" --out "$SSD/match_list.csv"
  echo ""
  echo "✅ 完成。相符的 .pat 在： $SSD/match_list.csv （matched=yes 的就是對上你病歷號的）"
  echo "   看相符清單： grep ',yes\$' $SSD/match_list.csv"
  exit 0
fi

# ---------- 階段 3：各 base 篩中 → 只下 cohort 完整(.sdb) ----------
for BASE in "${BASES[@]}"; do
  SL="$(slug "$BASE")"; D="$SSD/$SL"
  [ -s "$D/index.csv" ] || continue
  echo ""; echo "===== [cohort] $BASE 篩中清單 ====="
  python3 "$REPO/pipeline/filter_pats.py" --index "$D/index.csv" --cohort "$SSD/cohort.txt" \
      --out "$D/cohort_pats.txt"
  # 本機 pdb_only 絕對路徑 → 相對 base 路徑（給 mirror）
  sed "s#^${D}/pdb_only/##" "$D/cohort_pats.txt" | grep -v '^[[:space:]]*$' > "$D/cohort_rel.txt" || true
  if [ -s "$D/cohort_rel.txt" ]; then
    echo "----- 下載 cohort 完整(.sdb) -----"
    mkdir -p "$D/cohort_raw"
    bash "$REPO/pipeline/mirror_list.sh" "$D/cohort_rel.txt" "$BASE" "$D/cohort_raw" \
        "$FTP_HOST" "$FTP_USER" '' "$PAR" "$CHUNK_CHARSET"
  else
    echo "  此 base 無命中 cohort。"
  fi
done

# ---------- 階段 4：合併總 pat_log（帶 matched）+ 轉 h5 ----------
echo ""; echo "===== [全域] 合併總病歷號 log（含 matched）====="
python3 "$REPO/pipeline/pat_log.py" "${INDEX_ARGS[@]}" \
    --cohort "$SSD/cohort.txt" --out "$SSD/pat_log_all.csv"

if [ "$DO_H5" = "1" ]; then
  echo ""; echo "===== [全域] cohort 完整 → h5 ====="
  for BASE in "${BASES[@]}"; do
    SL="$(slug "$BASE")"; D="$SSD/$SL"
    [ -d "$D/cohort_raw" ] || continue
    ( cd "$REPO/pdb_to_h5" && python3 -m heyex_pipeline \
        --input "$D/cohort_raw" --output "$SSD/h5_output" \
        --workers "$WORKERS" --manifest-checkpoint-interval 10000 --verify-samples 5 )
  done
fi

echo ""
echo "✅ 完成。"
echo "  每 base log    : $SSD/<base>/pat_log.csv"
echo "  總 log         : $SSD/pat_log_all.csv   ← 每個 .pat 的 病歷號 + matched"
echo "  cohort 病歷號  : $SSD/cohort.txt"
echo "  cohort h5      : $SSD/h5_output"
echo "下一步（普查）: python3 -m forecast_c.census.a1_census --h5-dir $SSD/h5_output --out census_out"
