"""掃 .pat 下的 .pdb（32KB）→ index.csv（pat_dir, 病歷號）。只讀 .pdb、不解影像。

病歷號 = Heidelberg 患者記錄 surname 欄的數字（CGMH 慣例，如 "4561107-7" → 45611077）。
用於「先只下載 .pdb 篩 cohort、再只下載 cohort 完整 .pat」省掉 5TB 下載。

跑（容器內）:
  python pipeline/scan_pdb.py --input <含 .pat 的根目錄> --repo-root . --workers 16 --out index.csv
"""
import argparse
import csv
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor


def _setup_path(repo_root):
    """讓 worker 能 import 底層 Heidelberg parser。"""
    p = os.path.join(repo_root, "pdb_to_h5", "existing")
    if p not in sys.path:
        sys.path.insert(0, p)


def chart_from_pdb(pdb_path):
    """只開 .pdb → 取病人記錄(type 0x09)的 surname 數字 = 病歷號。失敗回 None。"""
    from export_e2e_csv import read_mdb_dir, parse_patient_data
    try:
        with open(pdb_path, "rb") as f:
            if f.read(4) != b"CMDb":
                return None
            f.seek(0x24)
            if f.read(7) != b"MDbMDir":
                return None
            for e in read_mdb_dir(f, 0x4c):
                if e["type"] == 0x09:                      # TYPE_PATIENT
                    pd = parse_patient_data(f, e)
                    digits = re.sub(r"\D", "", pd.get("surname", "") or "")
                    if digits:
                        return digits
    except Exception:
        return None
    return None


def _scan_one(pat_dir):
    pdbs = glob.glob(os.path.join(pat_dir, "*.pdb"))
    if not pdbs:
        return (pat_dir, "", "no_pdb")
    chart = chart_from_pdb(pdbs[0])
    return (pat_dir, chart or "", "ok" if chart else "no_chart")


def main():
    ap = argparse.ArgumentParser(description="掃 .pdb → 病歷號 index")
    ap.add_argument("--input", required=True, help="含 .pat 的根目錄（可只含 .pdb）")
    ap.add_argument("--repo-root", default=".", help="forecast-c repo 根（找 pdb_to_h5/existing）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="index.csv")
    a = ap.parse_args()

    _setup_path(a.repo_root)
    pats = sorted(glob.glob(os.path.join(a.input, "**", "*.pat"), recursive=True))
    print(f"找到 {len(pats)} 個 .pat，{a.workers} 核掃描中...")
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_setup_path,
                             initargs=(a.repo_root,)) as ex:
        for r in ex.map(_scan_one, pats, chunksize=64):
            rows.append(r)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pat_dir", "chart_no", "status"])
        w.writerows(rows)
    ok = sum(1 for r in rows if r[2] == "ok")
    print(f"寫出 {a.out}：{ok}/{len(rows)} 成功抽到病歷號"
          f"（no_pdb={sum(1 for r in rows if r[2]=='no_pdb')}, "
          f"no_chart={sum(1 for r in rows if r[2]=='no_chart')}）")


if __name__ == "__main__":
    main()
