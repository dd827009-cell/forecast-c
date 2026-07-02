#!/usr/bin/env bash
# 快速列出一個 base 底下的 .pat（相對 basename），用 cls 一次 LIST、不遞迴 → 很快。
# 給「直接 mirror -I '*.pdb' 抓所有 .pdb」用（不需先做慢的整棵樹遞迴列檔）。
#
# 用法:
#   bash list_pats.sh <REMOTE_BASE> <OUT_pat_list.txt> <HOST> <USER> [CHARSET=big5]
# 假設 .pat 是 base 的直接子目錄（本資料集如此）。若結構較深、列不到 .pat 會提示。
set -uo pipefail

BASE="${1:?給 REMOTE_BASE}"; OUT="${2:?給輸出清單路徑}"
HOST="${3:?給 HOST}"; USER="${4:?給 USER}"; CHARSET="${5:-big5}"
BASE="${BASE%/}"
mkdir -p "$(dirname "$OUT")"

# 快取：列過就別再列（大 base 的 cls 很慢）。要重列設 FORCE=1。
if [ "${FORCE:-0}" != "1" ] && [ -s "$OUT" ]; then
  echo "[list_pats] 已有 $OUT（$(grep -c . "$OUT") 個 .pat），略過重列。要重列設 FORCE=1。"
  exit 0
fi

# 加逾時 + 自動重連/重試：cls 連線 stall 不會永遠卡死；timeout 900 = 最多等 15 分鐘。
PRE="set ssl:verify-certificate no; set ftp:charset $CHARSET; set file:charset utf-8; \
set net:timeout 30; set net:max-retries 1; set net:reconnect-interval-base 15; \
set net:persist-retries 1; set cmd:interactive no;"
if ! timeout 900 lftp -u "$USER" "$HOST" -e "$PRE cls -1 '$BASE/'; bye" > "$OUT.raw" 2>"$OUT.err"; then
  echo "[list_pats] ⚠️ 連線/列檔失敗或逾時，見 $OUT.err（重跑會再試）"; exit 1
fi
# 取 .pat：去尾斜線、去路徑只留 basename
grep -iE '\.pat/?$' "$OUT.raw" | sed 's#/*$##; s#.*/##' | sort -u > "$OUT"
N=$(grep -c . "$OUT" 2>/dev/null || echo 0)
echo "[list_pats] $BASE → $N 個 .pat → $OUT"
if [ "$N" -eq 0 ]; then
  echo "[list_pats] ⚠️ 沒列到直接子層的 .pat（可能結構較深或不是 .pat 樹）。"
fi
