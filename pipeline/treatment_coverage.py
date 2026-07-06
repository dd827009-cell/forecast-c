#!/usr/bin/env python3
"""治療覆蓋分析：EYLEA 逐針治療 vs h5 cohort（補 census 缺的治療決策格）。

★ 自含版：內嵌 EYLEA 解析（只需 pandas），不 import forecast_c、不碰 torch。
   （治療條件的 tensor 化在 forecast_c/data/treatment.py，訓練時才需要 torch。）

用法: python pipeline/treatment_coverage.py <EYLEA.xlsx> <patient_stats.csv>
"""
import sys
import csv
import collections
import statistics
import datetime
import math

DRUG_DATE_KEY = {
    "aflibercept_8mg": "Aflibercept_8mg_施打日期", "aflibercept_2mg": "Aflibercept_2mg_施打日期",
    "bevacizumab": "Bevacizumab_施打日期", "ranibizumab": "Ranibizumab_施打日期",
    "faricimab": "Faricimab_施打日期", "brolucizumab": "Brolucizumab_施打日期",
    "dexamethasone": "Dexamethasone_施打日期", "other": "Other_施打日期",
}


def _parse_dates(cell):
    if cell is None:
        return []
    try:
        if isinstance(cell, float) and math.isnan(cell):
            return []
    except Exception:
        pass
    out = []
    for tok in str(cell).replace("、", ",").split(","):
        s = tok.strip().replace(".0", "")
        if len(s) == 8 and s.isdigit():
            try:
                out.append(datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8])))
            except ValueError:
                pass
    return out


def _find_col(cols, *keys):
    for c in cols:
        if any(k in str(c) for k in keys):
            return c
    return None


def parse_treatment(xlsx, sheet="重新整理過"):
    import pandas as pd
    df = pd.read_excel(xlsx, sheet_name=sheet)
    cols = list(df.columns)
    chart_c = _find_col(cols, "病歷號")
    eye_c = _find_col(cols, "OS(0)/OD(1)", "OD/OS")
    naive_c = _find_col(cols, "Naïve", "Naive")
    drug_cols = {d: _find_col(cols, k) for d, k in DRUG_DATE_KEY.items()}
    recs = {}
    for _, row in df.iterrows():
        chart = str(row[chart_c]).replace(".0", "").strip() if chart_c else ""
        if not chart.isdigit():
            continue
        eye = "OD" if (eye_c and str(row[eye_c]).replace(".0", "").strip() == "1") else "OS"
        events = []
        for d, col in drug_cols.items():
            if col is None:
                continue
            for dt in _parse_dates(row[col]):
                events.append((dt, d))
        events.sort()
        is_naive = bool(naive_c and str(row[naive_c]).replace(".0", "").strip() == "0")
        recs[(chart, eye)] = {"events": events, "is_naive": is_naive, "n_inject": len(events)}
    return recs


def main():
    if len(sys.argv) < 3:
        print("用法: python pipeline/treatment_coverage.py <EYLEA.xlsx> <patient_stats.csv>")
        sys.exit(1)
    eylea, stats = sys.argv[1], sys.argv[2]

    recs = parse_treatment(eylea)
    print(f"EYLEA 解析 {len(recs)} 條 (病歷號,eye) 治療軌跡")

    h5_eyes = set()
    for r in csv.DictReader(open(stats, encoding="utf-8-sig")):
        c = str(r.get("病歷號", "")).strip()
        if not c:
            continue
        if int(r.get("OD回診日數") or 0) > 0:
            h5_eyes.add((c, "OD"))
        if int(r.get("OS回診日數") or 0) > 0:
            h5_eyes.add((c, "OS"))

    tx_eyes = set(recs.keys())
    matched = h5_eyes & tx_eyes
    print(f"\nh5 有OCT的眼 = {len(h5_eyes)}")
    print(f"EYLEA 有治療的眼 = {len(tx_eyes)}")
    print(f"★ 對上(有OCT + 有EYLEA治療) = {len(matched)}  ← 能做『治療條件』預測的眼")

    ninj = collections.Counter(); drugs = collections.Counter(); naive = 0
    for k in matched:
        rec = recs[k]; ninj[rec["n_inject"]] += 1; naive += rec["is_naive"]
        for _, dn in rec["events"]:
            drugs[dn] += 1
    print(f"\n藥種變異(注射事件數): {dict(drugs.most_common())}")
    print("  → 用到 %d 種藥 → %s" % (
        len(drugs), "多種藥, Stage B 換藥可試" if len(drugs) > 1 else "單一藥, 只能比『有無治療』"))
    print(f"naive 眼數: {naive} / {len(matched)}")
    print(f"注射次數分布(對上的眼): {dict(sorted(ninj.items()))}")
    if matched:
        inj = [recs[k]["n_inject"] for k in matched]
        print(f"  中位注射次數 = {statistics.median(inj)}  最多 = {max(inj)}")


if __name__ == "__main__":
    main()
