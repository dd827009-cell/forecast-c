#!/usr/bin/env bash
# 對一個遠端 base 遞迴列檔 → listing.txt（每行一個檔案路徑）。只列名稱、不下載內容。
# 用途：(1) 算每個 .pat 的 .sdb 數（pat_log.py）；(2) 推出 .pat 清單給只下 .pdb 用。
#
# 為什麼不是一句 lftp find：一個 base 常有上千~上萬個 .pat，單線 find 很慢又看不到進度、
# 且輸出被緩衝（檔案看起來一直空的）。這裡改成：
#   Phase A：cls 一次列出 base 底下的 .pat（單一 LIST，很快）。
#   Phase B：對每個 .pat 平行 cls 它的內容（PAR 條連線），寫成各自的小檔，最後合併。
# → 平行、可續跑（已列的 .pat 跳過）、進度看得到（數 parts/ 裡的檔數）。
# 若 base 底下沒有直接的 .pat（結構較深）→ 自動退回單線 lftp find。
#
# 密碼走 ~/.netrc（不在本檔、不進 git）：
#   machine cad.csie.ntu.edu.tw login d13945010 password 你的密碼   → chmod 600 ~/.netrc
#
# 用法:
#   bash list_remote.sh <REMOTE_BASE> <OUT_listing.txt> <HOST> <USER> [CHARSET=big5] [PAR=6]
# 已存在且非空則略過（要重列：設 FORCE=1）。
set -uo pipefail

BASE="${1:?給 REMOTE_BASE}"; OUT="${2:?給輸出 listing 路徑}"
HOST="${3:?給 HOST}"; USER="${4:?給 USER}"; CHARSET="${5:-big5}"; PAR="${6:-6}"
BASE="${BASE%/}"
mkdir -p "$(dirname "$OUT")"

if [ "${FORCE:-0}" != "1" ] && [ -s "$OUT" ]; then
  echo "[list_remote] 已有 $OUT（$(wc -l < "$OUT") 行），略過。要重列設 FORCE=1。"
  exit 0
fi

PRE="set ssl:verify-certificate no; set ftp:charset $CHARSET; set file:charset utf-8; set net:connection-limit 2;"

echo "[list_remote] Phase A：列出 $BASE 底下的 .pat ..."
CHILDREN="$OUT.children"
if ! lftp -u "$USER" "$HOST" -e "$PRE cls -1 '$BASE/'; bye" > "$CHILDREN" 2>"$OUT.err"; then
  echo "[list_remote] ⚠️ 連線/列檔失敗，見 $OUT.err"; exit 1
fi
# 取出 .pat（去掉尾斜線、去路徑，只留 basename）；容大小寫
grep -iE '\.pat/?$' "$CHILDREN" | sed 's#/*$##; s#.*/##' | sort -u > "$OUT.pats"
NPAT=$(grep -c . "$OUT.pats" || echo 0)

if [ "$NPAT" -eq 0 ]; then
  echo "[list_remote] Phase A 沒找到直接的 .pat（結構較深）→ 退回單線 lftp find ..."
  if lftp -u "$USER" "$HOST" -e "$PRE find '$BASE'; bye" > "$OUT.tmp" 2>>"$OUT.err"; then
    mv "$OUT.tmp" "$OUT"; echo "[list_remote] 完成（find）：$(wc -l < "$OUT") 行 → $OUT"; exit 0
  else
    echo "[list_remote] ⚠️ find 也失敗，見 $OUT.err"; rm -f "$OUT.tmp"; exit 1
  fi
fi

echo "[list_remote] Phase B：平行列 $NPAT 個 .pat（$PAR 條連線，可續跑）..."
echo "               進度可另開視窗看： ls '$OUT.parts' | wc -l"
PARTS="$OUT.parts"; mkdir -p "$PARTS"
export HOST USER PRE BASE PARTS

list_one() {
  local pat="$1"
  [ -z "${pat// }" ] && return 0
  local key; key=$(printf '%s' "$pat" | md5sum | cut -c1-16)
  local f="$PARTS/$key"
  [ -s "$f" ] && return 0                                   # 續跑：已列過就跳
  if lftp -u "$USER" "$HOST" -e "$PRE cls -1 '$BASE/$pat'; bye" > "$f.tmp" 2>/dev/null; then
    mv "$f.tmp" "$f"
  else
    rm -f "$f.tmp"                                          # 失敗不留半檔 → 下次重試
  fi
}
export -f list_one

xargs -a "$OUT.pats" -P "$PAR" -I{} bash -c 'list_one "$@"' _ {}

done_n=$(find "$PARTS" -type f ! -name '*.tmp' | wc -l)
if [ "$done_n" -lt "$NPAT" ]; then
  echo "[list_remote] ⚠️ 只列到 $done_n/$NPAT 個 .pat（可能斷線）→ 重跑本指令會續列剩下的。"
fi
find "$PARTS" -type f ! -name '*.tmp' -exec cat {} + > "$OUT"
echo "[list_remote] 完成：$(wc -l < "$OUT") 行、$done_n/$NPAT 個 .pat → $OUT"
