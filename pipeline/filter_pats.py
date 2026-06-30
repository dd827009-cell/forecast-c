"""index.csv ∩ cohort.txt → cohort_pats.txt（要下載完整資料的 .pat 清單）+ 命中統計。

跑:
  python pipeline/filter_pats.py --index index.csv --cohort cohort.txt --out cohort_pats.txt
之後用 cohort_pats.txt 去 FTP 只下載這些 .pat 的完整資料（含大 .sdb），再轉 h5。
"""
import argparse
import csv


def main():
    ap = argparse.ArgumentParser(description="index ∩ cohort → 要下載的 .pat 清單")
    ap.add_argument("--index", required=True, help="scan_pdb 產的 index.csv")
    ap.add_argument("--cohort", required=True, help="cohort_list 產的 cohort.txt")
    ap.add_argument("--out", default="cohort_pats.txt")
    a = ap.parse_args()

    cohort = {line.strip() for line in open(a.cohort, encoding="utf-8") if line.strip()}
    keep, n_total, n_chart, hit_charts = [], 0, 0, set()
    with open(a.index, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_total += 1
            ch = row["chart_no"].strip()
            if ch:
                n_chart += 1
            if ch and ch in cohort:
                keep.append(row["pat_dir"])
                hit_charts.add(ch)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(keep))
    print(f"index {n_total} 筆（{n_chart} 有病歷號）∩ cohort {len(cohort)} 個")
    print(f"→ 命中 {len(keep)} 個 .pat（涵蓋 {len(hit_charts)} 個病歷號）→ {a.out}")
    if n_chart:
        print(f"  cohort 涵蓋率: {len(hit_charts)}/{len(cohort)} 個治療病歷號在這批資料裡有 OCT")


if __name__ == "__main__":
    main()
