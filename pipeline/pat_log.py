"""每個 .pat 一筆 log。可兩種來源（可只用一種、也可組合）：
  --index    scan_pdb.py 產的 index.csv（pat_dir, chart_no, status）→ 病歷號（主要）。
  --listing  remote 列檔（list_remote.sh 產）→ 每個 .pat 的 .sdb / .pdb / 檔數（選用）。
  --cohort   病歷號清單（cohort_list 產）→ 標記 matched。

**快路（只記 .pdb 病歷號）**：只給 --index 即可，不需要 --listing。
  python pat_log.py --index p=SSD/patients/index.csv --cohort SSD/cohort.txt --out SSD/pat_log.csv

join / 去重鍵 = .pat 的 basename。輸出 CSV 欄位:
  base, folder, pat_name, chart_no, n_sdb, n_pdb, n_files, matched
  沒有 --listing 時 n_sdb/n_pdb/n_files 留空（那份資訊要靠列檔才有）。
  matched: yes/no（有病歷號才判定；無病歷號留空）。
"""
import argparse
import csv
import os


def _split_label(spec, default_from):
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        label, path = os.path.splitext(os.path.basename(default_from or spec))[0], spec
    return label, path


def _folder_of(path):
    path = path.strip().rstrip("/")
    if not path:
        return None
    parts = path.split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower().endswith(".pat"):
            return "/".join(parts[: i + 1])
    return "/".join(parts[:-1]) if len(parts) > 1 else "(root)"


def main():
    ap = argparse.ArgumentParser(description="每個 .pat → 病歷號 (+ 選用 .sdb 數) log")
    ap.add_argument("--index", action="append", default=[],
                    help="[label=]index.csv（scan_pdb 產，含病歷號）。可重複。")
    ap.add_argument("--listing", action="append", default=[],
                    help="[label=]listing.txt（remote 列檔，選用，帶 .sdb 數）。可重複。")
    ap.add_argument("--cohort", default=None, help="病歷號清單（標 matched）。")
    ap.add_argument("--out", default="pat_log.csv")
    a = ap.parse_args()

    if not a.index and not a.listing:
        ap.error("至少要一份 --index 或 --listing")

    # 以 (base, .pat basename) 為鍵 → 不同 base 同名資料夾不會互相蓋掉。
    pats = {}  # (base,name) -> dict(base, folder, chart, n_sdb, n_pdb, n_files, has_counts)

    def rec(label, name):
        return pats.setdefault((label, name), {"base": label, "folder": "", "chart": "",
                                               "n_sdb": 0, "n_pdb": 0, "n_files": 0,
                                               "has_counts": False})

    # 列檔（選用）→ 每 .pat 的 .sdb/.pdb/檔數
    for spec in a.listing:
        label, path = _split_label(spec, None)
        n = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.rstrip("\n").rstrip("\r")
                if not line.strip():
                    continue
                n += 1
                folder = _folder_of(line)
                if folder is None:
                    continue
                name = os.path.basename(folder)
                r = rec(label, name)
                r["folder"] = r["folder"] or folder
                r["has_counts"] = True
                if line.endswith("/"):
                    continue
                ext = os.path.splitext(line)[1].lower()
                r["n_files"] += 1
                if ext == ".sdb":
                    r["n_sdb"] += 1
                elif ext == ".pdb":
                    r["n_pdb"] += 1
        print(f"  listing [{label}] {path}: {n} 行")

    # index → 病歷號（主要）
    for spec in a.index:
        label, path = _split_label(spec, None)
        n = 0
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pat_dir = (row.get("pat_dir") or "").strip().rstrip("/")
                if not pat_dir:
                    continue
                name = os.path.basename(pat_dir)
                r = rec(label, name)
                r["folder"] = r["folder"] or pat_dir
                chart = (row.get("chart_no") or "").strip()
                if chart:
                    if r["chart"] and r["chart"] != chart:
                        print(f"  ⚠️ {name} 有兩個病歷號 {r['chart']}/{chart}，取後者")
                    r["chart"] = chart
                    n += 1
        print(f"  index [{label}] {path}: {n} 筆有病歷號")

    cohort = set()
    if a.cohort:
        cohort = {x.strip() for x in open(a.cohort, encoding="utf-8") if x.strip()}
        print(f"  cohort {a.cohort}: {len(cohort)} 個病歷號")

    rows = []
    for (_label, name), r in pats.items():
        matched = ("yes" if r["chart"] in cohort else "no") if r["chart"] else ""
        n_sdb = r["n_sdb"] if r["has_counts"] else ""
        n_pdb = r["n_pdb"] if r["has_counts"] else ""
        n_files = r["n_files"] if r["has_counts"] else ""
        rows.append([r["base"], r["folder"], name, r["chart"],
                     n_sdb, n_pdb, n_files, matched])

    rows.sort(key=lambda x: (x[0], x[7] != "yes", str(x[3]), x[1]))

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base", "folder", "pat_name", "chart_no",
                    "n_sdb", "n_pdb", "n_files", "matched"])
        w.writerows(rows)

    n_chart = sum(1 for r in rows if r[3])
    n_match = sum(1 for r in rows if r[7] == "yes")
    matched_charts = {r[3] for r in rows if r[7] == "yes"}
    print(f"\n寫出 {a.out}：{len(rows)} 個 .pat")
    print(f"  有病歷號 {n_chart}；命中 cohort {n_match} 個 .pat（涵蓋 {len(matched_charts)} 個病歷號）")
    if cohort:
        print(f"  cohort 涵蓋率：{len(matched_charts)}/{len(cohort)} 個治療病歷號在這批資料裡有 OCT")


if __name__ == "__main__":
    main()
