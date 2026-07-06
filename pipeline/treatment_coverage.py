#!/usr/bin/env python3
"""治療覆蓋分析：EYLEA 逐針治療 vs h5 cohort（補 census 缺的治療決策格）。

在 forecast-c 根目錄跑：
  python pipeline/treatment_coverage.py <EYLEA.xlsx> <patient_stats.csv>

輸出：
  - 有 OCT 且有 EYLEA 治療紀錄的「眼」數（= 真正能做治療條件預測的訓練眼）
  - 藥種變異（→ Stage B 換藥可不可試）
  - naive 眼數 / 注射次數分布
"""
import sys
import csv
import collections
import statistics

from forecast_c.data.treatment import parse_treatment


def main():
    if len(sys.argv) < 3:
        print("用法: python pipeline/treatment_coverage.py <EYLEA.xlsx> <patient_stats.csv>")
        sys.exit(1)
    eylea, stats = sys.argv[1], sys.argv[2]

    recs = parse_treatment(eylea)                       # {(病歷號, eye): record}
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
        for _, dn, _ in rec["events"]:
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
