#!/usr/bin/env bash
# 批次鏡像一份「相對 .pat 路徑清單」（相對於 BASE）。可選只抓某種副檔名。
#
# ★關鍵：用「一條連線連抓一大批 .pat」（每個 lftp 連線跑 CHUNK 個 mirror 才登出），
#   而不是每個 .pat 開一條新連線。9 萬個 .pat 只登入數百次 → 又快、又不會撞 NAS 的
#   「每 IP 10 條連線」上限。這是 repo 舊版 download_pdb_batched.sh 的做法，這裡補上
#   中文 charset / 逾時 / 連線上限 / 已下載跳過。
#
# 兩階段共用：階段A 只抓 .pdb（GLOB='*.pdb'）；階段B 抓完整（GLOB 空，含大 .sdb）。
# 走 ~/.netrc（不碰明碼）。可續跑：已下載的 .pat 不再重抓。
#
# 用法:
#   bash mirror_list.sh <REL_LIST> <BASE> <OUT> <HOST> <USER> [GLOB] [PAR=3] [CHARSET=big5] [CHUNK=200]
set -uo pipefail

LIST="${1:?給 REL_LIST}"; BASE="${2:?給 BASE}"; OUT="${3:?給 OUT}"
HOST="${4:?給 HOST}"; USER="${5:?給 USER}"; GLOB="${6:-}"; PAR="${7:-3}"; CHARSET="${8:-big5}"; CHUNK="${9:-200}"
BASE="${BASE%/}"
mkdir -p "$OUT"
WORK="$OUT/_chunks"; mkdir -p "$WORK"; rm -f "$WORK"/chunk_* 2>/dev/null || true

# ⚠️ NAS 每 IP 上限 10 條連線 → 每條 lftp 連線 connection-limit 1；PAR 條並行 → 總連線 ≈ PAR。
PRE="set ssl:verify-certificate no; set ftp:charset $CHARSET; set file:charset utf-8; \
set net:connection-limit 1; set net:timeout 20; set net:max-retries 3; \
set net:reconnect-interval-base 5; set net:persist-retries 3; set cmd:interactive no; set xfer:clobber on;"
INC=""; [ -n "$GLOB" ] && INC="-I '$GLOB'"

# 1) 建 todo：跳過「已下載」的 .pat（pdb 模式已有 *.pdb、完整模式已有 *.sdb）。
TODO="$WORK/_todo.txt"; : > "$TODO"
total=0; skip=0
while IFS= read -r rel; do
  [ -z "${rel// }" ] && continue
  rel="${rel%/}"; total=$((total+1))
  dst="$OUT/$rel"
  if [ -z "$GLOB" ]; then
    ls "$dst"/*.sdb >/dev/null 2>&1 && { skip=$((skip+1)); continue; }
  elif ls "$dst"/$GLOB >/dev/null 2>&1; then
    skip=$((skip+1)); continue
  fi
  printf '%s\n' "$rel" >> "$TODO"
done < "$LIST"
n_todo=$(grep -c . "$TODO" 2>/dev/null || echo 0)
echo "[mirror_list] 共 $total 個 .pat；已下載略過 $skip；這次要抓 $n_todo（glob='${GLOB:-全部}'，$PAR 條連線、每批 $CHUNK）"
[ "$n_todo" -eq 0 ] && { echo "[mirror_list] 沒有要抓的，完成。"; exit 0; }

# 2) 切批
split -l "$CHUNK" -d -a 5 "$TODO" "$WORK/chunk_"

# 3) 每批用「一條連線」連抓（跑完整批才登出）
export OUT BASE HOST USER PRE INC GLOB
run_chunk() {
  local cf="$1" s pat
  s="$PRE"$'\n'
  while IFS= read -r pat; do
    [ -z "$pat" ] && continue
    mkdir -p "$OUT/$pat"
    s+="mirror $INC --continue \"$BASE/$pat\" \"$OUT/$pat\""$'\n'
  done < "$cf"
  s+="bye"$'\n'
  # 每批最多 2 小時（整批卡死的保險；正常遠比這快）。net:timeout 已處理單點 stall。
  timeout 7200 lftp -u "$USER" "$HOST" -e "$s" >/dev/null 2>&1 || true
}
export -f run_chunk

ls "$WORK"/chunk_* | xargs -P "$PAR" -I{} bash -c 'run_chunk "$@"' _ {}

# 4) 收尾：再數一次還沒下到的（＝失敗，重跑會續）
FAIL="$OUT/_failures.txt"; : > "$FAIL"
while IFS= read -r rel; do
  dst="$OUT/$rel"
  if [ -z "$GLOB" ]; then
    ls "$dst"/*.sdb >/dev/null 2>&1 || printf '%s\n' "$rel" >> "$FAIL"
  else
    ls "$dst"/$GLOB >/dev/null 2>&1 || printf '%s\n' "$rel" >> "$FAIL"
  fi
done < "$TODO"
n_fail=$(grep -c . "$FAIL" 2>/dev/null || echo 0)
echo "[mirror_list] 完成：這批 $((n_todo - n_fail))/$n_todo 成功（失敗 $n_fail 見 $FAIL，重跑本指令續抓）"
