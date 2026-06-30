"""治療 Excel → 病歷號清單 cohort.txt（給篩選用）。

掃 Excel 各 sheet 中欄名含「病歷號」的欄，抽數字。
⚠️ EYLEA 檔有「病歷號」欄（= h5 patient_id 同系統）；case-pooling 的「Chart no.」是別的序號，
   不含「病歷號」字樣 → 自然被跳過（正確）。

跑:
  python pipeline/cohort_list.py --excel "EYLEA....xlsx" ["全體系....xlsx"] --out cohort.txt
"""
import argparse
import re


def charts_from_excel(path):
    import pandas as pd
    out = set()
    xl = pd.ExcelFile(path)
    for sh in xl.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sh)
        except Exception:
            continue
        for c in df.columns:
            if "病歷號" in str(c):
                for x in df[c].dropna():
                    s = re.sub(r"\D", "", str(x))
                    if s:
                        out.add(s)
    return out


def main():
    ap = argparse.ArgumentParser(description="治療 Excel → 病歷號清單")
    ap.add_argument("--excel", nargs="+", required=True)
    ap.add_argument("--out", default="cohort.txt")
    a = ap.parse_args()

    charts = set()
    for x in a.excel:
        c = charts_from_excel(x)
        print(f"  {x}: 抽到 {len(c)} 個病歷號")
        charts |= c
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(charts)))
    print(f"共 {len(charts)} 個唯一病歷號 → {a.out}")


if __name__ == "__main__":
    main()
