#!/usr/bin/env python3
"""對帳：OCT(855眼/441病人) vs 細治療(EYLEA) vs 粗治療(case pooling)。
釐清「363粗+34細 對不上 441」= 眼vs病人單位不同 + 部分眼/病人無治療紀錄。

在 forecast-c 根目錄跑（需 pandas 讀 EYLEA）：
  python pipeline/treatment_reconcile.py <patient_stats.csv> <EYLEA.xlsx> <case_pooling.xlsx>
"""
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 讓 import case_pooling 生效
from case_pooling import parse_case_pooling  # noqa: E402

EYLEA_DRUG = ["Aflibercept_8mg_施打日期", "Aflibercept_2mg_施打日期", "Bevacizumab_施打日期",
              "Ranibizumab_施打日期", "Faricimab_施打日期", "Brolucizumab_施打日期",
              "Dexamethasone_施打日期", "Other_施打日期"]


def eylea_eyes_with_injection(xlsx):
    """EYLEA『重新整理過』→ 有 ≥1 針的 (病歷號,eye) 集合。"""
    import pandas as pd
    df = pd.read_excel(xlsx, sheet_name="重新整理過")
    cols = list(df.columns)

    def fc(*ks):
        for c in cols:
            if any(k in str(c) for k in ks):
                return c
        return None

    cc = fc("病歷號"); ec = fc("OS(0)/OD(1)", "OD/OS")
    dcs = [fc(k) for k in EYLEA_DRUG]
    out = set()
    for _, row in df.iterrows():
        ch = str(row[cc]).replace(".0", "").strip() if cc else ""
        if not ch.isdigit():
            continue
        n = 0
        for c in dcs:
            if c is None:
                continue
            for t in str(row[c]).replace("、", ",").split(","):
                s = t.strip().replace(".0", "")
                if len(s) == 8 and s.isdigit():
                    n += 1
        if n > 0:
            eye = "OD" if (ec and str(row[ec]).replace(".0", "").strip() == "1") else "OS"
            out.add((ch, eye))
    return out


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    stats, eylea, cp = sys.argv[1], sys.argv[2], sys.argv[3]

    oct_eyes, oct_pat = set(), set()
    for r in csv.DictReader(open(stats, encoding="utf-8-sig")):
        c = str(r.get("病歷號", "")).strip()
        if not c:
            continue
        if int(r.get("OD回診日數") or 0) > 0:
            oct_eyes.add((c, "OD")); oct_pat.add(c)
        if int(r.get("OS回診日數") or 0) > 0:
            oct_eyes.add((c, "OS")); oct_pat.add(c)

    fine = eylea_eyes_with_injection(eylea) & oct_eyes
    coarse = set(parse_case_pooling(cp)) & oct_eyes
    either = fine | coarse

    print("=== 眼層級 ===")
    print(f"OCT 眼 = {len(oct_eyes)}")
    print(f"  細治療(EYLEA) = {len(fine)}")
    print(f"  粗治療(case pooling) = {len(coarse)}")
    print(f"  兩者都有 = {len(fine & coarse)}")
    print(f"  有任一治療 = {len(either)}")
    print(f"  無任何治療紀錄 = {len(oct_eyes - either)}")

    pf = {c for c, e in fine}; pc = {c for c, e in coarse}; pe = pf | pc
    print("\n=== 病人層級 ===")
    print(f"OCT 病人 = {len(oct_pat)}")
    print(f"  有細治療 = {len(pf)}")
    print(f"  有粗治療 = {len(pc)}")
    print(f"  有任一治療 = {len(pe)}")
    print(f"  無任何治療紀錄 = {len(oct_pat - pe)}")

    no_tx = sorted(oct_pat - pe)
    print(f"\n無治療紀錄的病人（前 20）: {no_tx[:20]}")
    with open("/mnt/d/oct_no_treatment_charts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(no_tx))
    print("完整清單 → D:\\oct_no_treatment_charts.txt")


if __name__ == "__main__":
    main()
