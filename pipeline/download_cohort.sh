#!/usr/bin/env bash
# 只下載 cohort 的完整 .pat（含大 .sdb）。被 run_nas_pipeline.sh 呼叫，也可單獨用。
# 密碼走 ~/.netrc（不在本檔），lftp 自動讀。可續跑：已下載完整的略過。
#
# 用法:
#   bash download_cohort.sh <cohort_pats.txt> <REMOTE_BASE> <OUT目錄> <HOST> <USER> <PDB_ROOT>
set -euo pipefail

LIST="$1"; REMOTE_BASE="$2"; OUT="$3"; HOST="$4"; USER="$5"; PDB_ROOT="$6"
mkdir -p "$OUT"
: > "$OUT/_failures.txt"

n_total=$(grep -c . "$LIST" || true)
n=0
while IFS= read -r p; do
    [ -z "$p" ] && continue
    n=$((n+1))
    rel="${p#"$PDB_ROOT"/}"                 # 本機 pdb_only 路徑 → 遠端相對路徑
    dst="$OUT/$rel"
    # 續跑：若已有 .sdb（代表完整下載過）就略過
    if ls "$dst"/*.sdb >/dev/null 2>&1; then
        echo "[$n/$n_total] 跳過(已有) $rel"; continue
    fi
    echo "[$n/$n_total] 下載 $rel ..."
    mkdir -p "$dst"
    if ! lftp -u "$USER" "$HOST" -e "set net:connection-limit 4; mirror --parallel=4 --continue '$REMOTE_BASE/$rel' '$dst'; bye"; then
        echo "$rel" >> "$OUT/_failures.txt"
        echo "  ⚠️ 失敗，記錄到 _failures.txt"
    fi
done < "$LIST"

n_fail=$(grep -c . "$OUT/_failures.txt" || true)
echo "cohort 下載完成：$((n - n_fail))/$n 成功（失敗 $n_fail 見 $OUT/_failures.txt，可重跑本腳本續抓）"
