#!/usr/bin/env python3
"""比對治療 cohort(Excel 病歷號) vs h5 輸出統計(patient_stats.csv)。純標準庫、零安裝。

輸出:
  matched_cohort.csv  = 治療 cohort 且有 OCT h5（= Phase 2 可用 cohort，含 OD/OS 回診數）
  missing_cohort.txt  = 治療 cohort 但這批 h5 裡沒有（沒下到 / 沒 OCT）
  extra_h5.txt        = 有 h5 但不在治療 cohort（非治療病人）

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

    cohort = set()
    for x in excels:
        c = charts_from_xlsx(x)
        print(f"  {os.path.basename(x)}: {len(c)} 個病歷號")
        cohort |= c
    print(f"cohort 唯一病歷號 = {len(cohort)}")

    stats = {}
    with open(stats_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            stats[str(r.get("病歷號", "")).strip()] = r
    h5set = set(k for k in stats if k)
    print(f"h5 輸出病人數 = {len(h5set)}")

    matched = cohort & h5set
    missing = cohort - h5set
    extra = h5set - cohort
    print(f"\n★ 對得上(cohort 且有 OCT h5) = {len(matched)}")
    print(f"  cohort 但這批無 OCT h5     = {len(missing)}")
    print(f"  有 h5 但不在 cohort        = {len(extra)}")
    if cohort:
        print(f"  cohort 涵蓋率 = {len(matched)}/{len(cohort)} = {len(matched)/len(cohort):.1%}")

    with open(f"{outdir}/matched_cohort.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["病歷號", "OD回診日數", "OS回診日數", "總就診日", "首日", "末日"])
        for c in sorted(matched):
            r = stats[c]
            w.writerow([c, r.get("OD回診日數"), r.get("OS回診日數"),
                        r.get("總就診日"), r.get("首日"), r.get("末日")])
    with open(f"{outdir}/missing_cohort.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(missing)))
    with open(f"{outdir}/extra_h5.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(extra)))
    print(f"\n寫出 → {outdir}/matched_cohort.csv（Phase2 可用 cohort）, missing_cohort.txt, extra_h5.txt")


if __name__ == "__main__":
    main()
