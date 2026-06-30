#!/usr/bin/env bash
# 分批下載大量 .pat 內的 .pdb，避免單一 mirror 卡在「列 10 萬個目錄」整棵樹。
# 作法：先用 cls 拿到 .pat 名單，再「每 CHUNK 個 .pat 用一條 lftp 連線跑多個 mirror」，
#       PAR 條連線並行 → count 會穩定往上爬、可續跑（已下載的 .pdb 會被 mirror 跳過）。
#
# 用法:
#   bash download_pdb_batched.sh <pat_list.txt> <REMOTE_BASE> <OUT> <HOST> [CHUNK] [PAR]
# 例:
#   bash download_pdb_batched.sh /mnt/c/octdata/big/pat_list.txt \
#       "/eye2/eye4(cad5)/ike/patients" /mnt/c/octdata/big/pdb_only \
#       cad.csie.ntu.edu.tw 300 6
set -uo pipefail

LIST="$1"; BASE="$2"; OUT="$3"; HOST="$4"; CHUNK="${5:-300}"; PAR="${6:-4}"
PRE="set ssl:verify-certificate no; set net:connection-limit 2;"
mkdir -p "$OUT"
WORK="$OUT/_chunks"; mkdir -p "$WORK"; rm -f "$WORK"/chunk_* "$WORK/_names.txt"

# .pat 基名（去路徑、去尾斜線），只留 .pat
sed 's#/*$##; s#.*/##' "$LIST" | grep '\.pat$' > "$WORK/_names.txt"
N=$(wc -l < "$WORK/_names.txt"); echo "共 $N 個 .pat，每批 $CHUNK、並行 $PAR 連線"
split -l "$CHUNK" -d -a 5 "$WORK/_names.txt" "$WORK/chunk_"

export OUT BASE HOST PRE
run_chunk() {
  local cf="$1" s="$PRE"$'\n' pat
  while IFS= read -r pat; do
    [ -z "$pat" ] && continue
    s+="mirror -I '*.pdb' --continue \"$BASE/$pat\" \"$OUT/$pat\""$'\n'
  done < "$cf"
  s+="bye"$'\n'
  lftp "$HOST" -e "$s" >/dev/null 2>&1 || true
}
export -f run_chunk

ls "$WORK"/chunk_* | xargs -P "$PAR" -I{} bash -c 'run_chunk "$@"' _ {}
echo "完成。.pdb 總數: $(find "$OUT" -name '*.pdb' | wc -l)"
