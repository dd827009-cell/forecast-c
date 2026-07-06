#!/usr/bin/env python3
"""找出「可補下的治療病人」：治療病歷號 ∩ 全NAS.pdb − 已有OCT。
= 在 NAS 有 .pdb、但完整 .pat/OCT 還沒下/沒轉的治療病人 → 補下他們就能擴大治療 cohort。

在 forecast-c 根跑：
  python pipeline/recoverable_tx.py <nas_full_index.csv> <patient_stats.csv> <EYLEA.xlsx> <case_pooling.xlsx>
"""
import sys
import os
import csv
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_pdb_match import read_sheet, charts_from_sheet  # noqa: E402


def main():
    if len(sys.argv) < 5:
        print(__doc__); sys.exit(1)
    nas_idx, stats, eylea, cp = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    # 全 NAS: 病歷號 → 該病歷號的 .pat 路徑(可能多個/跨來源)
    chart2pat = collections.defaultdict(list)
    for r in csv.DictReader(open(nas_idx, encoding="utf-8")):
        c = str(r.get("chart_no", "")).strip()
        if c:
            chart2pat[c].append(r.get("pat_dir", ""))
    nas_charts = set(chart2pat)

    oct_charts = set()
    for r in csv.DictReader(open(stats, encoding="utf-8-sig")):
        c = str(r.get("病歷號", "")).strip()
        if c:
            oct_charts.add(c)

    tx = set()
    for path, sheet, col, hkey in [(eylea, "重新整理過", None, "病歷號"),
                                    (eylea, "AI整理", "A", None),
                                    (eylea, "nAMD", "A", None),
                                    (cp, "data collection", "D", None)]:
        rows = read_sheet(path, sheet)
        ch, _ = charts_from_sheet(rows, col_letter=col, header_key=hkey)
        tx |= ch

    tx_in_nas = tx & nas_charts
    tx_has_oct = tx & oct_charts
    recoverable = tx_in_nas - oct_charts       # 在NAS、有治療、但還沒轉成OCT

    print(f"治療病歷號 = {len(tx)}")
    print(f"  其中在全 NAS 有 .pdb = {len(tx_in_nas)}")
    print(f"  其中已有 OCT h5      = {len(tx_has_oct)}")
    print(f"  ★ 可補下(在NAS、未轉OCT) = {len(recoverable)}")
    print(f"  NAS 也沒有(補不了)   = {len(tx - nas_charts)}")

    pat_lines = []
    for c in sorted(recoverable):
        for p in chart2pat[c]:
            if p:
                pat_lines.append(p)
    with open("/mnt/d/recoverable_charts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(recoverable)))
    with open("/mnt/d/recoverable_pats.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(pat_lines))
    print(f"\n可補下病歷號 → D:\\recoverable_charts.txt ({len(recoverable)})")
    print(f"對應 .pat 路徑 → D:\\recoverable_pats.txt ({len(pat_lines)})")


if __name__ == "__main__":
    main()
