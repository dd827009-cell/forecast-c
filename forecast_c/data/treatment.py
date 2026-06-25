"""治療紀錄 → 模型條件（TreatmentEncoder dict）。

把臨床 Excel（EYLEA 8mg「重新整理過」sheet：逐針藥種+日期）轉成 ForecastModel 的治療條件。

連結鍵: **病歷號 ↔ h5 `patient_id`** + **OS/OD ↔ h5 `laterality`** + 施打日期 ↔ h5 `acquisition_time_utc`。
  （EYLEA 病歷號 = h5 patient_id 同 ID 系統；case pooling 的「Chart no.」是別的序號，勿用。）

藥種 id（anti-VEGF 學名 ↔ 商品名）:
  aflibercept_8mg=Eylea 8mg / aflibercept_2mg=Eylea 2mg / bevacizumab=Avastin /
  ranibizumab=Lucentis / faricimab=Vabysmo / brolucizumab=Beovu / dexamethasone / other。0=padding。

對某次 OCT 回診（病歷號 P, eye E, 日期 D）:
  drug_ids  : 該眼在 D 之前最近 M 次注射的藥種 id
  numerics  : 每事件 [距該針天數(到 D), 第幾針(序), 累積針數]
  is_naive  : Naïve(0)→True
  → 餵 TreatmentEncoder；另回 baseline 嚴重度標記(CRT/IRF/...)與變乾結果(存活頭)。
"""
import datetime

import numpy as np
import torch

# 藥種學名 → id（1..N；0 保留 padding）。商品名見模組 docstring。
DRUG_IDS = {
    "aflibercept_8mg": 1, "aflibercept_2mg": 2, "bevacizumab": 3, "ranibizumab": 4,
    "faricimab": 5, "brolucizumab": 6, "dexamethasone": 7, "other": 8,
}
# 各藥「施打日期」欄的關鍵字（Excel 欄名以此 substring 比對，容錯空白/變體）
DRUG_DATE_KEY = {
    "aflibercept_8mg": "Aflibercept_8mg_施打日期", "aflibercept_2mg": "Aflibercept_2mg_施打日期",
    "bevacizumab": "Bevacizumab_施打日期", "ranibizumab": "Ranibizumab_施打日期",
    "faricimab": "Faricimab_施打日期", "brolucizumab": "Brolucizumab_施打日期",
    "dexamethasone": "Dexamethasone_施打日期", "other": "Other_施打日期",
}


def _parse_dates(cell):
    """'20250903, 20251119' → [date, ...]（容錯空白/非數字）。"""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
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
        cs = str(c)
        if any(k in cs for k in keys):
            return c
    return None


def parse_treatment(xlsx_path, sheet="重新整理過"):
    """Excel → {(病歷號, eye): record}。eye ∈ {'OD','OS'}。"""
    import pandas as pd
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    cols = list(df.columns)
    chart_c = _find_col(cols, "病歷號")
    eye_c = _find_col(cols, "OS(0)/OD(1)", "OD/OS")
    naive_c = _find_col(cols, "Naïve", "Naive")
    drug_cols = {d: _find_col(cols, k) for d, k in DRUG_DATE_KEY.items()}

    records = {}
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
                events.append((dt, d, DRUG_IDS[d]))
        events.sort(key=lambda e: e[0])
        is_naive = bool(naive_c and str(row[naive_c]).replace(".0", "").strip() == "0")
        records[(chart, eye)] = {"chart": chart, "eye": eye, "events": events,
                                 "is_naive": is_naive, "n_inject": len(events)}
    return records


def _as_date(d):
    if isinstance(d, datetime.date):
        return d
    s = str(d)[:10]
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(s.replace("-", "")[:8], "%Y%m%d").date()
        except ValueError:
            continue
    return None


def treatment_dict_at(record, visit_date, max_events=8, numeric_in=3):
    """某次回診（visit_date）→ TreatmentEncoder dict（batch=1 tensors）。

    只取 visit_date 之前（含當天）的注射；最近 max_events 個事件餵 drug_ids/numerics。
    numerics 每事件 = [距該針天數, 第幾針(1-based), 累積針數]。
    """
    vd = _as_date(visit_date)
    past = [(d, dn, did) for (d, dn, did) in record["events"] if vd is None or d <= vd]
    past.sort(key=lambda e: e[0])
    total = len(past)
    recent = past[-max_events:]                                   # 最近 M 個

    M = max_events
    drug_ids = np.zeros((1, M), np.int64)
    numerics = np.zeros((1, M, numeric_in), np.float32)
    for j, (d, dn, did) in enumerate(recent):
        ordinal = total - len(recent) + j + 1                     # 第幾針（全序列）
        days_since = (vd - d).days if vd is not None else 0
        drug_ids[0, j] = did
        numerics[0, j, :min(3, numeric_in)] = [days_since, ordinal, ordinal][:numeric_in]
    return {
        "drug_ids": torch.from_numpy(drug_ids),
        "numerics": torch.from_numpy(numerics),
        "event_mask": torch.from_numpy((drug_ids > 0)),
        "is_naive": torch.tensor([record["is_naive"]]),
        "n_inject_to_date": total,
    }


# ───────────────────────── self-test（需 EYLEA Excel）`python -m forecast_c.data.treatment <xlsx>` ─────────────────────────
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    path = sys.argv[1] if len(sys.argv) > 1 else "/dl/EYLEA  8mg 恩慈整理完成 (3).xlsx"
    recs = parse_treatment(path)
    print(f"解析 {len(recs)} 條 (病歷號,eye) 治療軌跡")
    # 找一條有多針的
    rec = max(recs.values(), key=lambda r: r["n_inject"])
    print(f"  範例 病歷號={rec['chart']} eye={rec['eye']} 總針數={rec['n_inject']} naive={rec['is_naive']}")
    drugs = sorted(set(dn for _, dn, _ in rec["events"]))
    print(f"  用過的藥: {drugs}")
    # 在「最後一針之後」當回診 → drug_ids/numerics
    last_date = rec["events"][-1][0]
    td = treatment_dict_at(rec, last_date, max_events=8)
    print(f"  視為回診@{last_date}: drug_ids={td['drug_ids'].tolist()[0]}")
    print(f"    numerics[0]= [距針天數,序,累積] 前3事件: {td['numerics'][0,:3].tolist()}")
    print(f"    n_inject_to_date={td['n_inject_to_date']}, is_naive={bool(td['is_naive'][0])}")
    assert td["drug_ids"].shape == (1, 8) and td["numerics"].shape == (1, 8, 3)
    assert td["event_mask"].sum() > 0
    print("treatment loader self-test 通過 ✅（Excel 逐針 → TreatmentEncoder dict）")
