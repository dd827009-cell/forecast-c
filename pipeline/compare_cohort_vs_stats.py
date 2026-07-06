#!/usr/bin/env python3
"""比對治療 cohort(Excel 病歷號) vs h5 輸出統計(patient_stats.csv)。純標準庫、零安裝。

逐份 Excel 分開報（各自有多少人有 OCT）+ 合併 + 兩份重疊。輸出:
  matched_cohort.csv  = 任一 Excel 且有 OCT h5（= Phase 2 可用 cohort，含 OD/OS 回診數）
  missing_cohort.txt  = cohort 但這批 h5 裡沒有
  extra_h5.txt        = 有 h5 但不在任何 cohort

用法:
  python compare_cohort_vs_stats.py <patient_stats.csv> <out_dir> <excel1.xlsx> [excel2.xlsx ...]
"""
import sys
import re
import csv
import os
import zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CHART = re.compile(r"^\d{7,8}$")


def _shared_strings(z):
    try:
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(t.text or "" for t in si.iter(f"{NS}t")) for si in root.findall(f"{NS}si")]


def charts_from_xlsx(path):
    found = set()
    with zipfile.ZipFile(path) as z:
        ss = _shared_strings(z)
        for sn in [n for n in z.namelist() if n.startswith("xl/worksheets/") and n.endswith(".xml")]:
            for c in ET.fromstring(z.read(sn)).iter(f"{NS}c"):
                t = c.get("t"); v = c.find(f"{NS}v")
                if v is not None and v.text is not None:
                    val = ss[int(v.text)] if (t == "s" and v.text.isdigit()) else v.text
                else:
                    is_ = c.find(f"{NS}is")
                    val = "".join(tt.text or "" for tt in is_.iter(f"{NS}t")) if is_ is not None else None
                if val is None:
                    continue
                val = str(val).strip()
                if val.endswith(".0") and val[:-2].isdigit():
                    val = val[:-2]
                if CHART.match(val):
                    found.add(val)
    return found


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    stats_csv, outdir = sys.argv[1], sys.argv[2]
    excels = sys.argv[3:]
    os.makedirs(outdir, exist_ok=True)

    stats = {}
    with open(stats_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            stats[str(r.get("病歷號", "")).strip()] = r
    h5set = set(k for k in stats if k)
    print(f"h5 輸出病人數 = {len(h5set)}\n")

    # ---- 逐份 Excel 分開報 ----
    per = {}
    print("=== 逐份 Excel（各自有多少人有 OCT）===")
    for x in excels:
        c = charts_from_xlsx(x)
        per[x] = c
        m = c & h5set
        cov = f"{len(m)}/{len(c)} = {len(m)/len(c):.1%}" if c else "0"
        print(f"  {os.path.basename(x)}")
        print(f"    病歷號 {len(c)}｜有 OCT {len(m)}｜涵蓋 {cov}")

    # ---- 兩份重疊 ----
    if len(excels) == 2:
        a, b = per[excels[0]], per[excels[1]]
        print(f"\n兩份 Excel 共同病歷號 = {len(a & b)}"
              f"（{os.path.basename(excels[0])} 獨有 {len(a - b)}｜"
              f"{os.path.basename(excels[1])} 獨有 {len(b - a)}）")

    # ---- 合併 ----
    cohort = set().union(*per.values()) if per else set()
    matched = cohort & h5set
    missing = cohort - h5set
    extra = h5set - cohort
    print(f"\n=== 合併(任一 Excel) ===")
    print(f"cohort 唯一病歷號 = {len(cohort)}")
    print(f"★ 對上(有 OCT h5) = {len(matched)}")
    print(f"  cohort 但無 OCT = {len(missing)}")
    print(f"  有 OCT 但不在 cohort = {len(extra)}")

    with open(f"{outdir}/matched_cohort.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["病歷號", "OD回診日數", "OS回診日數", "總就診日", "首日", "末日",
                    "在_" + os.path.basename(excels[0])] + (["在_" + os.path.basename(excels[1])] if len(excels) == 2 else []))
        for c in sorted(matched):
            r = stats[c]
            flags = [("Y" if c in per[x] else "") for x in excels]
            w.writerow([c, r.get("OD回診日數"), r.get("OS回診日數"),
                        r.get("總就診日"), r.get("首日"), r.get("末日")] + flags)
    with open(f"{outdir}/missing_cohort.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(missing)))
    with open(f"{outdir}/extra_h5.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(extra)))
    print(f"\n寫出 → {outdir}/matched_cohort.csv（含每人來自哪份 Excel）, missing_cohort.txt, extra_h5.txt")


if __name__ == "__main__":
    main()
