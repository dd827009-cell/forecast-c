#!/usr/bin/env bash
# 下載 cohort 的「完整 .pat」（含 .sdb 影像）。用驗證過可行的 lftp 寫法：bare host + ssl no。
# 可續跑（已有 .sdb 的 .pat 直接略過）、可並行。
#
# 用法: bash fetch_cohort_full.sh <cohort_pats.txt> <REMOTE_BASE> <OUT> <HOST> [PAR]
# 例:
#   bash fetch_cohort_full.sh /mnt/c/octdata/big/cohort_pats.txt \
#       "/eye2/eye4(cad5)/ike/patients" /mnt/c/octdata/big/cohort_raw \
#       cad.csie.ntu.edu.tw 4
set -uo pipefail

LIST="$1"; BASE="$2"; OUT="$3"; HOST="$4"; PAR="${5:-4}"
mkdir -p "$OUT"
export OUT BASE HOST

fetch_one() {
  local p="$1" name
  name="$(basename "$p")"
  [ -z "$name" ] && return 0
  ls "$OUT/$name"/*.sdb >/dev/null 2>&1 && return 0   # 已完整 → 略過
  lftp "$HOST" -e "set ssl:verify-certificate no; mirror --parallel=4 --continue \"$BASE/$name\" \"$OUT/$name\"; bye" >/dev/null 2>&1 || true
}
export -f fetch_one

grep -v '^[[:space:]]*$' "$LIST" | xargs -P "$PAR" -I{} bash -c 'fetch_one "$@"' _ {}
echo "完成。cohort 完整資料夾數: $(find "$OUT" -maxdepth 1 -name '*.pat' | wc -l)"
