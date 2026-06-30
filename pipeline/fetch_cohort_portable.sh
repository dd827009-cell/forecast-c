#!/usr/bin/env bash
# 可攜版：在任何有 lftp 的機器下載 cohort 的完整 .pat（含 .sdb 影像）。
# 不需 repo、不需 Python。只要：lftp + 這支腳本 + cohort_pats.txt + NAS 帳密。
#
# 【帳密】不要寫進腳本！在這台建立 ~/.netrc：
#     machine cad.csie.ntu.edu.tw login d13945010 password 你的密碼
#   再 chmod 600 ~/.netrc
#   （lftp 會自動讀取；本腳本完全不碰明碼，也不會進 git。）
#
# 用法:
#   bash fetch_cohort_portable.sh <cohort_pats.txt> <輸出目錄> [並行數=4]
# 例:
#   bash fetch_cohort_portable.sh cohort_pats.txt ./cohort_raw 4
#
# 特性：可續跑（已有 .sdb 的 .pat 自動略過）、並行、印每筆進度、最後印完成數。
set -uo pipefail

# ===== NAS 設定（非機密，可放這裡）=====
HOST="cad.csie.ntu.edu.tw"
BASE="/eye2/eye4(cad5)/ike/patients"
# =======================================

LIST="${1:?請給 cohort_pats.txt 路徑}"
OUT="${2:?請給輸出目錄}"
PAR="${3:-4}"
mkdir -p "$OUT"
export OUT BASE HOST

total=$(grep -vc '^[[:space:]]*$' "$LIST" || true)
echo "要下載 $total 個 .pat → $OUT （並行 $PAR 連線）"

fetch_one() {
  local p="$1" name
  name="$(basename "$p")"; [ -z "$name" ] && return 0
  if ls "$OUT/$name"/*.sdb >/dev/null 2>&1; then
    echo "[skip] $name"; return 0
  fi
  if lftp "$HOST" -e "set ssl:verify-certificate no; mirror --parallel=4 --continue \"$BASE/$name\" \"$OUT/$name\"; bye" >/dev/null 2>&1; then
    echo "[ok]   $name"
  else
    echo "[FAIL] $name"
  fi
}
export -f fetch_one

grep -v '^[[:space:]]*$' "$LIST" | xargs -P "$PAR" -I{} bash -c 'fetch_one "$@"' _ {}

done_n=$(find "$OUT" -name "*.sdb" 2>/dev/null | sed 's#/[^/]*$##' | sort -u | wc -l)
echo "完成。已完整(.sdb)資料夾數: $done_n / $total"
