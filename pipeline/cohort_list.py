"""治療 Excel → 病歷號清單 cohort.txt（給篩選用）。

兩條抽取規則（聯集）：
  規則1（EYLEA）：欄名含「病歷號」→ 抽該欄全部數字。
  規則2（case pooling）：任何『純 7–8 位整數』的儲存格 → 候選病歷號。
     case pooling 去識別化後把病歷號填進「Name」欄（無「病歷號」欄名），格式就是 7–8 位醫院病歷號；
     研究序號(Chart no.)只有 4 位、CMT/VA/Excel 日期序號都不是 7–8 位 → 規則2 安全。
  最終仍由 filter_pats 與 .pdb 掃出的 index.csv 取交集驗證，非病歷號的 7–8 位雜訊自然落空。

跑:
  python pipeline/cohort_list.py --excel "EYLEA....xlsx" ["全體系....xlsx"] --out cohort.txt
"""
import argparse
import re

_CHART = re.compile(r"\d{7,8}")


def charts_from_excel(path):
    import pandas as pd
    out, n_named, n_digit = set(), 0, 0
    xl = pd.ExcelFile(path)
    for sh in xl.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sh)
        except Exception:
            continue
        for c in df.columns:
            named = "病歷號" in str(c)
            for x in df[c].dropna():
                # 數值欄常被 pandas 讀成 float（1008842.0）→ 還原整數字串
                if isinstance(x, float) and x.is_integer():
                    s = str(int(x))
                else:
                    s = str(x).strip()
                if named:                          # 規則1
                    d = re.sub(r"\D", "", s)
                    if d:
                        out.add(d); n_named += 1
                elif _CHART.fullmatch(s):          # 規則2：純 7–8 位整數
                    out.add(s); n_digit += 1
    return out, n_named, n_digit


def main():
    ap = argparse.ArgumentParser(description="治療 Excel → 病歷號清單")
    ap.add_argument("--excel", nargs="+", required=True)
    ap.add_argument("--out", default="cohort.txt")
    a = ap.parse_args()

    charts = set()
    for x in a.excel:
        c, n_named, n_digit = charts_from_excel(x)
        print(f"  {x}: 抽到 {len(c)} 個病歷號（病歷號欄 {n_named} 格 / 7-8位整數 {n_digit} 格）")
        charts |= c
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(charts)))
    print(f"共 {len(charts)} 個唯一病歷號 → {a.out}")


if __name__ == "__main__":
    main()
