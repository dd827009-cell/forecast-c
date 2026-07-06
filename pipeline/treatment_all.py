#!/usr/bin/env python3
"""整合所有治療來源的覆蓋分析：
  細治療(逐針日期): EYLEA「重新整理過」(有眼) + 「AI整理」(無眼→指派到病人OCT眼)
  粗治療(藥種/施打日): case pooling(有眼) + EYLEA「nAMD」W欄施打日(有眼,跳過'?')
vs h5 OCT(855眼/441病人)。補回只在 EYLEA 原始分頁的病人。

在 forecast-c 根跑(需 pandas 讀 EYLEA)：
  python pipeline/treatment_all.py <patient_stats.csv> <EYLEA.xlsx> <case_pooling.xlsx>
"""
import sys
import os
import csv
import math
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case_pooling import parse_case_pooling  # noqa: E402

DRUG_DATE = ["Aflibercept_8mg_施打日期", "Aflibercept_2mg_施打日期", "Bevacizumab_施打日期",
             "Ranibizumab_施打日期", "Faricimab_施打日期", "Brolucizumab_施打日期",
             "Dexamethasone_施打日期", "Other_施打日期"]


def _ndates(cell):
    try:
        if isinstance(cell, float) and math.isnan(cell):
            return 0
    except Exception:
        pass
    return sum(1 for t in str(cell).replace("、", ",").split(",")
               if len(t.strip().replace(".0", "")) == 8 and t.strip().replace(".0", "").isdigit())


def _find(cols, *ks):
    for c in cols:
        if any(k in str(c) for k in ks):
            return c
    return None


def _chart_ok(ch):
    return ch.isdigit() and 7 <= len(ch) <= 8


def eylea_reorg(xlsx):
    import pandas as pd
    df = pd.read_excel(xlsx, sheet_name="重新整理過"); cols = list(df.columns)
    cc = _find(cols, "病歷號"); ec = _find(cols, "OS(0)/OD(1)", "OD/OS")
    dcs = [_find(cols, k) for k in DRUG_DATE]
    out = set()
    for _, r in df.iterrows():
        ch = str(r[cc]).replace(".0", "").strip() if cc else ""
        if not _chart_ok(ch) or sum(_ndates(r[c]) for c in dcs if c is not None) == 0:
            continue
        eye = "OD" if (ec and str(r[ec]).replace(".0", "").strip() == "1") else "OS"
        out.add((ch, eye))
    return out


def eylea_ai(xlsx):
    import pandas as pd
    df = pd.read_excel(xlsx, sheet_name="AI整理"); cols = list(df.columns)
    cc = cols[0]  # A 欄 = 病歷號(表頭「整」)
    dcs = [_find(cols, k) for k in DRUG_DATE]
    out = set()
    for _, r in df.iterrows():
        ch = str(r[cc]).replace(".0", "").strip()
        if _chart_ok(ch) and sum(_ndates(r[c]) for c in dcs if c is not None) > 0:
            out.add(ch)
    return out


def eylea_namd(xlsx):
    import pandas as pd
    df = pd.read_excel(xlsx, sheet_name="nAMD"); cols = list(df.columns)
    cc = _find(cols, "病歷號"); ec = _find(cols, "OD/OS"); dc = _find(cols, "施打日")
    out = set()
    for _, r in df.iterrows():
        ch = str(r[cc]).replace(".0", "").strip() if cc else ""
        if not _chart_ok(ch):
            continue
        ev = str(r[ec]).replace(".0", "").strip() if ec else ""
        if ev not in ("0", "1"):
            continue
        eye = "OD" if ev == "1" else "OS"
        if dc and _ndates(r[dc]) > 0:
            out.add((ch, eye))
    return out


def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)
    stats, eylea, cp = sys.argv[1], sys.argv[2], sys.argv[3]

    oct_eyes = set(); oct_pat = collections.defaultdict(set)
    for r in csv.DictReader(open(stats, encoding="utf-8-sig")):
        c = str(r.get("病歷號", "")).strip()
        if not c:
            continue
        if int(r.get("OD回診日數") or 0) > 0:
            oct_eyes.add((c, "OD")); oct_pat[c].add("OD")
        if int(r.get("OS回診日數") or 0) > 0:
            oct_eyes.add((c, "OS")); oct_pat[c].add("OS")

    reorg = eylea_reorg(eylea)
    ai = eylea_ai(eylea)
    namd = eylea_namd(eylea)
    cpp = set(parse_case_pooling(cp))

    ai_assigned = set()
    for ch in ai:
        for eye in oct_pat.get(ch, ()):
            ai_assigned.add((ch, eye))

    fine = (reorg & oct_eyes) | ai_assigned
    coarse = ((cpp & oct_eyes) | (namd & oct_eyes)) - fine
    either = fine | coarse

    print("=== 各來源 ∩ OCT眼 ===")
    print(f"  重新整理過(細,有眼): {len(reorg & oct_eyes)}")
    print(f"  AI整理(細,病人指派): {len(ai_assigned)}")
    print(f"  nAMD(粗,有眼): {len(namd & oct_eyes)}")
    print(f"  case pooling(粗,有眼): {len(cpp & oct_eyes)}")
    print("\n=== 合併 眼層級 ===")
    print(f"OCT 眼 = {len(oct_eyes)}")
    print(f"  細治療 = {len(fine)}   粗治療 = {len(coarse)}   有任一 = {len(either)}   無治療 = {len(oct_eyes - either)}")
    pf = {c for c, e in fine}; pe = {c for c, e in either}
    print("\n=== 合併 病人層級 ===")
    print(f"OCT 病人 = {len(oct_pat)}")
    print(f"  有細治療 = {len(pf)}   有任一治療 = {len(pe)}   無治療 = {len(set(oct_pat) - pe)}")


if __name__ == "__main__":
    main()
