"""A-1 普查 driver — 規格 `solutions/A1_census_spec.md`。

燒 L40 前用純 CPU 算五組數字 + 決策表，決定 Stage A 可不可行 / 要不要降級。
設計: **計算邏輯（compute_census）與 I/O（build_trajectories_from_manifest）分離**
  → 計算層可用合成軌跡 self-test（現在就能驗），I/O 層等真資料接 manifest + M7b npz。

五組數字:
  ① 變乾事件數      → 要不要存活頭        （recovery.py）
  ② 軌跡長度分布    → 一步 / 多步 rollout
  ③ 治療分布        → 治療消融 / Stage B 可行性
  ④ 會變的眼比例    → persistence 招牌立不立得住
  ⑤ 次數 vs 時間 r  → 「第幾次」claim 是否保守
  （附）Δt 分布

跑（合成 self-test）: python forecast_c/census/a1_census.py --selftest
跑（真資料）       : python -m forecast_c.census.a1_census --manifest stage0/manifest.parquet \
                       --m7b-dir <dir> [--treatment <csv>] [--pilot N] --out census_out/
"""
import argparse
import json
import os

import numpy as np

try:                                  # 兩種跑法都支援: `-m forecast_c.census.a1_census` 或直接執行檔案
    from .recovery import recovery_rate
    from .cst import cst_from_npz
except ImportError:                   # 直接執行檔案（無 parent package）時的後備
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from forecast_c.census.recovery import recovery_rate
    from forecast_c.census.cst import cst_from_npz

# 決策門檻（規格 §1 / §2 決策表）
DRY_THRESHOLD_UM = 300.0          # 暫定乾閾值（待醫師分病種定案）
DRY_EVENT_MIN = 100              # 變乾眼 <100 → 純軌跡（不做存活頭）
LONG_TRAJ_MIN_VISITS = 3         # ≥3 visit 才有多步 rollout 意義
CHANGE_STABLE_UM = 25.0          # 末次−baseline |ΔCST| < 25 → 穩定
CHANGE_LARGE_UM = 75.0           # ≥75 → 大變；之間 = 輕變
CHANGING_EYE_MIN_FRAC = 0.30     # 會變眼比例 < 30% → persistence 招牌存疑
COUNT_TIME_R_HIGH = 0.90         # r>0.9 → 「第幾次」與時間糾纏，claim 要保守


# --------------------------------------------------------------------------- #
# 計算層（純 numpy / stdlib，可用合成軌跡驗）
# trajectory dict 結構（每眼一條）:
#   cst        : [float, ...]   每 visit 中央 CST（µm，可含 NaN）
#   dt_days    : [float, ...]   相鄰 visit 間隔天數（len = n_visit-1）；無則 []
#   drugs      : [str|int, ...] 每次注射藥種（無治療紀錄則 []）
#   is_naive   : bool           treatment-naive（對照）
#   split      : str            'train'/'val'/'test'（④ 用 train 算變化基準也行；此處全算）
# --------------------------------------------------------------------------- #
def compute_census(trajectories,
                   dry_threshold_um=DRY_THRESHOLD_UM,
                   change_stable_um=CHANGE_STABLE_UM,
                   change_large_um=CHANGE_LARGE_UM):
    """list[trajectory] → 五組數字 dict（+ 決策表）。"""
    cst_seqs = [t["cst"] for t in trajectories]
    n_eyes = len(trajectories)

    # ① 變乾事件數
    rec = recovery_rate(cst_seqs, dry_threshold_um)

    # ② 軌跡長度分布
    lengths = np.array([len(c) for c in cst_seqs], dtype=int)
    n_long = int((lengths >= LONG_TRAJ_MIN_VISITS).sum())
    traj = {
        "median_visits": float(np.median(lengths)) if n_eyes else 0.0,
        "max_visits": int(lengths.max()) if n_eyes else 0,
        "n_ge3_visits": n_long,
        "frac_ge3_visits": (n_long / n_eyes) if n_eyes else 0.0,
        "hist": _hist(lengths, bins=range(1, int(lengths.max()) + 2)) if n_eyes else {},
    }

    # ③ 治療分布
    has_tx = any(t.get("drugs") for t in trajectories)
    if has_tx:
        drug_eyes, inj_counts = {}, []
        n_naive = 0
        for t in trajectories:
            drugs = t.get("drugs") or []
            inj_counts.append(len(drugs))
            for d in set(drugs):
                drug_eyes[str(d)] = drug_eyes.get(str(d), 0) + 1
            if t.get("is_naive") or len(drugs) == 0:
                n_naive += 1
        inj = np.array(inj_counts, dtype=int)
        treatment = {
            "available": True,
            "n_drug_types": len(drug_eyes),
            "eyes_per_drug": drug_eyes,
            "inj_count_median": float(np.median(inj)) if inj.size else 0.0,
            "inj_count_max": int(inj.max()) if inj.size else 0,
            "naive_frac": (n_naive / n_eyes) if n_eyes else 0.0,
        }
    else:
        treatment = {"available": False,
                     "note": "無治療 metadata → 只跑軌跡/變乾部分；補治療紀錄列 R-1 待辦"}

    # ④ 會變的眼比例（末次 − baseline |ΔCST|）
    deltas = []
    for c in cst_seqs:
        finite = [v for v in c if v is not None and np.isfinite(v)]
        if len(finite) >= 2:
            deltas.append(abs(finite[-1] - finite[0]))
    deltas = np.array(deltas, dtype=float)
    n_d = deltas.size
    n_stable = int((deltas < change_stable_um).sum())
    n_large = int((deltas >= change_large_um).sum())
    n_mild = n_d - n_stable - n_large
    frac_changing = ((n_mild + n_large) / n_d) if n_d else 0.0
    changing = {
        "n_eyes_with_2plus_visits": int(n_d),
        "frac_stable": (n_stable / n_d) if n_d else 0.0,
        "frac_mild": (n_mild / n_d) if n_d else 0.0,
        "frac_large": (n_large / n_d) if n_d else 0.0,
        "frac_changing": frac_changing,
    }

    # ⑤ 次數 vs 時間 可識別性（累積注射次數 vs 累積 Δt 天）
    if has_tx:
        cum_counts, cum_days = [], []
        for t in trajectories:
            drugs = t.get("drugs") or []
            dts = t.get("dt_days") or []
            # 對每個「轉移」(visit i→i+1) 取 (累積到 i+1 的注射數, 累積 Δt)
            cumd = np.cumsum([0.0] + list(dts))
            for i in range(1, len(cumd)):
                cum_days.append(float(cumd[i]))
                cum_counts.append(float(min(i, len(drugs))))   # 近似: 第 i 次轉移≈打了 i 針
        r = _safe_corr(cum_counts, cum_days)
        count_time = {"available": True, "pearson_r": r,
                      "n_points": len(cum_days),
                      "identifiable": (r is not None and abs(r) < COUNT_TIME_R_HIGH)}
    else:
        count_time = {"available": False}

    # （附）Δt 分布（相鄰 visit 間隔天數）
    all_dt = [d for t in trajectories for d in (t.get("dt_days") or [])]
    all_dt = np.array(all_dt, dtype=float)
    dt_dist = ({"median_days": float(np.median(all_dt)), "min_days": float(all_dt.min()),
                "max_days": float(all_dt.max()), "n_intervals": int(all_dt.size)}
               if all_dt.size else {"n_intervals": 0})

    census = {
        "n_eyes": n_eyes,
        "recovery": rec,
        "trajectory": traj,
        "treatment": treatment,
        "changing": changing,
        "count_time": count_time,
        "dt": dt_dist,
        "thresholds": {"dry_threshold_um": dry_threshold_um,
                       "change_stable_um": change_stable_um,
                       "change_large_um": change_large_um},
    }
    census["decisions"] = decision_table(census)
    return census


def decision_table(c):
    """依五組數字自動填決策表（規格 §2）。回傳 list[{item, value, threshold, conclusion}]。"""
    rows = []
    n_dry = c["recovery"]["n_dry"]
    rows.append({"item": "變乾事件數", "value": n_dry, "threshold": f"<{DRY_EVENT_MIN}/≥{DRY_EVENT_MIN}",
                 "conclusion": "純軌跡（不做存活頭）" if n_dry < DRY_EVENT_MIN else "完整 Stage A（含存活頭）"})

    frac3 = c["trajectory"]["frac_ge3_visits"]
    rows.append({"item": "軌跡長度", "value": f"median={c['trajectory']['median_visits']}, ≥3={frac3:.0%}",
                 "threshold": "多數=2 / 有 ≥3",
                 "conclusion": "一步預測（V1）" if frac3 < 0.5 else "多步 rollout + SSM"})

    tx = c["treatment"]
    if tx["available"]:
        single = tx["n_drug_types"] <= 1
        rows.append({"item": "藥種變異", "value": f"{tx['n_drug_types']} 種",
                     "threshold": "單一 / 多種",
                     "conclusion": "Stage B 換藥不可（只比有無治療）" if single else "Stage B 換藥可試"})
    else:
        rows.append({"item": "藥種變異", "value": "無治療 metadata", "threshold": "—",
                     "conclusion": "補治療紀錄（R-1）後再評"})

    fc = c["changing"]["frac_changing"]
    rows.append({"item": "會變眼比例", "value": f"{fc:.0%}", "threshold": f"<{CHANGING_EYE_MIN_FRAC:.0%}/≥",
                 "conclusion": "重評目標（persistence 難立）" if fc < CHANGING_EYE_MIN_FRAC else "招牌可立"})

    ct = c["count_time"]
    if ct.get("available") and ct.get("pearson_r") is not None:
        rows.append({"item": "次數-時間相關", "value": f"r={ct['pearson_r']:.2f}",
                     "threshold": f"r>{COUNT_TIME_R_HIGH}/<",
                     "conclusion": "「第幾次」保守" if abs(ct["pearson_r"]) >= COUNT_TIME_R_HIGH else "次數/時間可分離"})
    else:
        rows.append({"item": "次數-時間相關", "value": "N/A（無治療紀錄）", "threshold": "—", "conclusion": "—"})
    return rows


def _hist(arr, bins):
    counts, edges = np.histogram(np.asarray(arr), bins=list(bins))
    return {int(edges[i]): int(counts[i]) for i in range(len(counts))}


def _safe_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------------------------------------------------------- #
# I/O 層（接真資料；需 pandas + M7b npz）— 等資料到位驗
# --------------------------------------------------------------------------- #
def build_trajectories_from_manifest(manifest_path, m7b_dir, treatment_path=None,
                                     pilot=None, npz_key_col="key", time_col="acquisition_time_utc"):
    """讀 manifest.parquet + M7b per-visit npz → list[trajectory]。

    需求欄位: patient_id, eye, longitudinal_key, visit_id, {time_col}, split, {npz_key_col}。
    M7b npz: <m7b_dir>/<key>.npz（含 ilm/rpe/ilm_valid/rpe_valid + meta）。
    treatment_path: 可選 CSV/parquet，欄位 (longitudinal_key, visit_id, drug, ...)；無則治療欄留空。
    """
    import pandas as pd

    df = pd.read_parquet(manifest_path)
    tx = None
    if treatment_path and os.path.exists(treatment_path):
        tx = (pd.read_parquet(treatment_path) if treatment_path.endswith(".parquet")
              else pd.read_csv(treatment_path))

    keys = list(df["longitudinal_key"].unique())
    if pilot:
        keys = keys[:pilot]

    trajectories = []
    for key in keys:
        g = df[df["longitudinal_key"] == key].sort_values(time_col)
        cst_seq, dt_seq, prev_t = [], [], None
        for _, row in g.iterrows():
            npz = os.path.join(m7b_dir, f"{row[npz_key_col]}.npz")
            cst = float("nan")
            if os.path.exists(npz):
                _, cst, _ = cst_from_npz(npz)
            cst_seq.append(cst)
            t = pd.to_datetime(row[time_col], errors="coerce")
            if prev_t is not None and t is not None:
                dt_seq.append(float((t - prev_t).days))
            prev_t = t
        drugs = []
        if tx is not None:
            drugs = list(tx[tx["longitudinal_key"] == key].get("drug", []))
        trajectories.append({
            "longitudinal_key": str(key),
            "eye": str(g.iloc[0].get("eye", "")),
            "patient_id": str(g.iloc[0].get("patient_id", "")),
            "split": str(g.iloc[0].get("split", "")),
            "cst": cst_seq, "dt_days": dt_seq,
            "drugs": drugs, "is_naive": len(drugs) == 0,
        })
    return trajectories


def _f(v, default=None):
    try:
        v = float(v)
        return v if (v == v and v > 0) else default
    except (TypeError, ValueError):
        return default


def _dt_days(iso_times):
    """ISO 時間字串列 → 相鄰間隔天數列（len = n-1）。無法解析則跳過。"""
    from datetime import datetime
    ds = []
    for t in iso_times:
        try:
            ds.append(datetime.fromisoformat(str(t).replace("Z", "+00:00")))
        except (ValueError, TypeError):
            ds.append(None)
    out = []
    for i in range(1, len(ds)):
        if ds[i] and ds[i - 1]:
            out.append(abs((ds[i] - ds[i - 1]).days))
    return out


def _visit_day(iso_str):
    """ISO 時間字串 → 'YYYY-MM-DD'（同日判定用）；無法解析回 None。"""
    s = str(iso_str or "")
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return None


def dedup_sameday(records):
    """同一眼「同一天多次掃描」去重：同日只留 quality 最高那筆。

    records: list[dict]，每筆需含 'day'(str|None) 與 'quality'(float)。
    day=None（無時間戳、無法判定同日）的各自保留，不誤併。
    回傳 (deduped_list, n_removed)。

    ★ 這是「同日重掃」去重；跨天的回診全留（縱向資料）。完全重複的掃描在轉檔階段
      已由 (病歷號,日時分秒,眼) 命名 + 冪等擋掉。未來 build_dataloader 實作時應 import
      本函式沿用同一規則，避免訓練/普查不一致。
    """
    by_day, keep_unknown = {}, []
    for r in records:
        d = r.get("day")
        if d is None:
            keep_unknown.append(r)
            continue
        cur = by_day.get(d)
        if cur is None or r.get("quality", float("-inf")) > cur.get("quality", float("-inf")):
            by_day[d] = r
    deduped = list(by_day.values()) + keep_unknown
    return deduped, len(records) - len(deduped)


def build_trajectories_from_h5(h5_dir, pilot=None, dedup_sameday_visits=True):
    """讀 h5_output（stage0 study 檔）→ (list[trajectory], dedup_info)。

    依 longitudinal_key=病歷號::眼 分組、按時間排序、每 visit 算 CST。
    去重: 同一眼同一天多次掃描（同日重掃）→ 預設只留品質最高那筆；跨天回診全留（縱向資料）。
    品質指標: image_quality_per_bscan 中位數（無則用 valid_ascan_mask 比例）。
    無治療 metadata（h5 不含）→ drugs 留空、is_naive=True（census 自動只跑軌跡/變乾部分）。
    """
    import glob
    import h5py
    from .cst import central_subfield_thickness, DEFAULT_AXIAL_UM_PER_PX

    files = sorted(glob.glob(os.path.join(h5_dir, "**", "*.h5"), recursive=True))
    groups = {}
    for f in files:
        with h5py.File(f, "r") as h:
            a = dict(h.attrs)
            ilm, rpe, valid = h["ilm_y"][:], h["rpe_bm_y"][:], h["valid_ascan_mask"][:]
            if "image_quality_per_bscan" in h:
                iq = h["image_quality_per_bscan"][:]
                q = float(np.nanmedian(iq)) if np.isfinite(iq).any() else float(valid.mean())
            else:
                q = float(valid.mean())
        cst, _ = central_subfield_thickness(
            ilm, rpe, axial_um_per_px=_f(a.get("scale_axial_um_per_px"), DEFAULT_AXIAL_UM_PER_PX),
            lateral_mm_per_px=_f(a.get("scale_lateral_mm_per_px")),
            bscan_spacing_mm=_f(a.get("scale_bscan_spacing_mm")),
            ilm_valid=valid, rpe_valid=valid)
        t = str(a.get("acquisition_time_utc", ""))
        key = str(a["longitudinal_key"])
        groups.setdefault(key, []).append(
            {"time": t, "day": _visit_day(t), "cst": cst, "quality": q,
             "lat": str(a.get("laterality", "")), "pid": str(a.get("patient_id", ""))})

    keys = list(groups)[:pilot] if pilot else list(groups)
    trajectories = []
    n_raw = n_removed = n_eyes_sameday = 0
    for key in keys:
        recs = groups[key]
        n_raw += len(recs)
        if dedup_sameday_visits:
            recs, n_rm = dedup_sameday(recs)
            n_removed += n_rm
            n_eyes_sameday += (n_rm > 0)
        visits = sorted(recs, key=lambda x: x["time"])        # 按時間排序
        trajectories.append({
            "longitudinal_key": key, "patient_id": visits[0]["pid"], "eye": visits[0]["lat"],
            "split": "", "cst": [v["cst"] for v in visits],
            "dt_days": _dt_days([v["time"] for v in visits]),
            "drugs": [], "is_naive": True})
    dedup_info = {"enabled": dedup_sameday_visits, "n_raw_visits": n_raw,
                  "n_after_dedup": n_raw - n_removed, "n_sameday_removed": n_removed,
                  "n_eyes_with_sameday": int(n_eyes_sameday)}
    return trajectories, dedup_info


def render_markdown(c):
    """census dict → 人讀 census_report.md（含決策表）。"""
    L = ["# A-1 普查報告\n", f"- 眼數: **{c['n_eyes']}**"]
    if c.get("dedup", {}).get("enabled"):
        d = c["dedup"]
        L.append(f"- 同日重掃去重: 原始 {d['n_raw_visits']} 次掃描 → 去重後 **{d['n_after_dedup']}** 次 visit"
                 f"（移除同日重複 {d['n_sameday_removed']} 筆；{d['n_eyes_with_sameday']} 眼有同日重掃，留品質最高那筆）")
    L += [f"- 變乾眼: {c['recovery']['n_dry']}（變乾率 {c['recovery']['dry_rate']:.1%}）",
         f"- 軌跡長度: median={c['trajectory']['median_visits']}, "
         f"≥3 visit {c['trajectory']['frac_ge3_visits']:.1%}, max={c['trajectory']['max_visits']}",
         f"- 會變眼比例: {c['changing']['frac_changing']:.1%} "
         f"(穩定 {c['changing']['frac_stable']:.0%} / 輕變 {c['changing']['frac_mild']:.0%} / 大變 {c['changing']['frac_large']:.0%})"]
    if c["treatment"]["available"]:
        L.append(f"- 治療: {c['treatment']['n_drug_types']} 種藥, "
                 f"naive {c['treatment']['naive_frac']:.0%}, 注射數 median {c['treatment']['inj_count_median']}")
    else:
        L.append(f"- 治療: {c['treatment'].get('note', '無')}")
    if c["count_time"].get("available") and c["count_time"].get("pearson_r") is not None:
        L.append(f"- 次數-時間相關 r = {c['count_time']['pearson_r']:.3f}")
    L.append("\n## 決策表\n")
    L.append("| 普查結果 | 數值 | 門檻 | 結論 |")
    L.append("|---|---|---|---|")
    for r in c["decisions"]:
        L.append(f"| {r['item']} | {r['value']} | {r['threshold']} | {r['conclusion']} |")
    return "\n".join(L) + "\n"


def write_reports(c, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "census_report.json"), "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "census_report.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(c))
    print(f"[census] 寫出 {out_dir}/census_report.{{json,md}}")


# --------------------------------------------------------------------------- #
# self-test: python forecast_c/census/a1_census.py --selftest
# --------------------------------------------------------------------------- #
def _self_test():
    print("[self-test] a1_census: 合成軌跡 → 五組數字 + 決策表 ...")
    rng = np.random.default_rng(0)
    trajectories = []
    for i in range(50):
        n_visit = int(rng.integers(2, 5))
        base = float(rng.uniform(300, 500))
        # 一半的眼會明顯下降（變乾），一半穩定
        if i % 2 == 0:
            cst = list(np.linspace(base, base - 180, n_visit))      # 大變，可能變乾
        else:
            cst = list(base + rng.normal(0, 5, n_visit))            # 穩定
        dt = list(rng.uniform(25, 45, n_visit - 1))
        drugs = ["antiVEGF"] * (n_visit - 1)
        trajectories.append({"cst": cst, "dt_days": dt, "drugs": drugs,
                             "is_naive": False, "split": "train"})

    c = compute_census(trajectories)
    assert c["n_eyes"] == 50
    assert c["recovery"]["n_dry"] > 0, "應有變乾眼"
    assert 0 <= c["changing"]["frac_changing"] <= 1
    assert c["trajectory"]["max_visits"] <= 4
    assert c["treatment"]["available"] and c["treatment"]["n_drug_types"] == 1
    assert len(c["decisions"]) == 5
    md = render_markdown(c)
    assert "決策表" in md and "變乾事件數" in md
    print(f"  [OK] 變乾眼={c['recovery']['n_dry']}, 會變={c['changing']['frac_changing']:.0%}, "
          f"r={c['count_time'].get('pearson_r')}")
    print(f"  [OK] 決策表 {len(c['decisions'])} 列、markdown 渲染成功")

    # 無治療 metadata 分支
    c2 = compute_census([{"cst": [400, 350], "dt_days": [30], "drugs": [], "is_naive": True}])
    assert not c2["treatment"]["available"] and not c2["count_time"]["available"]
    print("  [OK] 無治療 metadata 分支正常降級")

    # 同日重掃去重
    assert _visit_day("2021-03-15T10:30:00") == "2021-03-15" and _visit_day("") is None
    recs = [
        {"day": "2021-03-15", "quality": 0.8, "time": "2021-03-15T10:00:00", "tag": "a"},
        {"day": "2021-03-15", "quality": 0.95, "time": "2021-03-15T10:20:00", "tag": "b"},  # 同日、品質較高 → 留這個
        {"day": "2021-04-20", "quality": 0.7, "time": "2021-04-20T09:00:00", "tag": "c"},   # 不同天 → 留
        {"day": None, "quality": 0.5, "time": "", "tag": "d"},                              # 無時間戳 → 保留
    ]
    dd, n_rm = dedup_sameday(recs)
    assert n_rm == 1 and len(dd) == 3
    kept = {r["tag"] for r in dd}
    assert kept == {"b", "c", "d"}, kept
    print(f"  [OK] 同日重掃去重: 4→3（移除 {n_rm}，同日留品質高者 b、跨天 c 全留、無戳 d 保留）")
    print("[self-test OK]")


def main():
    ap = argparse.ArgumentParser(description="A-1 普查（設計 C）")
    ap.add_argument("--selftest", action="store_true", help="跑合成資料 self-test（不需真資料）")
    ap.add_argument("--h5-dir", help="h5_output 目錄（stage0 study 檔；直接讀，免 manifest）")
    ap.add_argument("--manifest", help="stage0/manifest.parquet")
    ap.add_argument("--m7b-dir", help="M7b per-visit npz 目錄")
    ap.add_argument("--treatment", help="治療 metadata CSV/parquet（可選）")
    ap.add_argument("--pilot", type=int, default=None, help="只跑前 N 條軌跡（pilot 驗邏輯）")
    ap.add_argument("--out", default="census_out", help="輸出目錄")
    a = ap.parse_args()

    dedup_info = None
    if a.h5_dir:
        trajectories, dedup_info = build_trajectories_from_h5(a.h5_dir, pilot=a.pilot)
    elif a.manifest:
        trajectories = build_trajectories_from_manifest(a.manifest, a.m7b_dir, a.treatment, pilot=a.pilot)
    else:
        _self_test(); return
    census = compute_census(trajectories)
    if dedup_info is not None:
        census["dedup"] = dedup_info
    write_reports(census, a.out)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
