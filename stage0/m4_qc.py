"""
Stage 0 - Module 4: QC 過濾 (volume-level)。

依 index.parquet 的欄位做品質過濾, 標記每顆 volume 是否通過 (qc_pass) 與失敗原因 (qc_flags)。
不刪列, 全部保留 + 加旗標, 供 M6 patient-split 只在 qc_pass==True 上切, 並保留可追溯性。

硬性丟棄規則 (任一觸發即 qc_pass=False):
  - read_error        : M1 讀檔錯誤 (error 非空)
  - no_volume         : 無 volume 資料 (vol_d 為空)
  - low_quality       : image_quality < --min-quality (預設 15, Heidelberg Q-score)
  - low_valid_ratio   : valid_ascan_ratio < --min-valid-ratio (預設 0.9)
  - wrong_n_bscans    : vol_d != --n-bscans (預設 25; n_bscans attr 同步檢查)
  - width_inconsistent: w_consistent == False (附帶陣列 W != volume W, 會破壞 M3 對齊)

軟性旗標 (僅記錄, 不丟棄, 寫入 qc_soft):
  - no_layers         : 缺 ilm_y/rpe_bm_y (僅影響 Stage2 層健檢, MAE 訓練仍可用)
  - no_valid_mask     : 缺 valid_ascan_mask (M2 門檻統計退化為全像素, 影響極小)

可選 --check-bscans: 讀每顆 (passing) volume 的 image_quality_per_bscan,
  另記 min_bscan_q / n_bad_bscans (個別壞 slice), 供日後 per-slice 處理; 預設關 (省 IO)。

用法:
    python stage0/m4_qc.py --index stage0/index.parquet --out stage0/index_qc.parquet
    python stage0/m4_qc.py --check-bscans            # 另讀 h5 算 per-bscan 品質
"""
import argparse
import os

import numpy as np
import pandas as pd


def compute_qc(df, min_quality, min_valid_ratio, n_bscans):
    """回傳 (qc_flags_list, qc_soft_list); 每列一個 ';' 串接的旗標字串 (無則空字串)。"""
    hard, soft = [], []
    for _, r in df.iterrows():
        h, s = [], []
        # 硬性
        if pd.notna(r.get("error")):
            h.append("read_error")
        if pd.isna(r.get("vol_d")):
            h.append("no_volume")
        else:
            if pd.notna(r.get("image_quality")) and r["image_quality"] < min_quality:
                h.append("low_quality")
            if pd.notna(r.get("valid_ascan_ratio")) and r["valid_ascan_ratio"] < min_valid_ratio:
                h.append("low_valid_ratio")
            if int(r["vol_d"]) != n_bscans or (
                    pd.notna(r.get("n_bscans")) and int(r["n_bscans"]) != n_bscans):
                h.append("wrong_n_bscans")
            if r.get("w_consistent") is False:
                h.append("width_inconsistent")
        # 軟性
        if not (bool(r.get("has_ilm")) and bool(r.get("has_rpe_bm"))):
            s.append("no_layers")
        if not bool(r.get("has_valid_mask")):
            s.append("no_valid_mask")
        hard.append(";".join(h))
        soft.append(";".join(s))
    return hard, soft


def add_bscan_quality(df, min_quality):
    """讀每列 h5 的 image_quality_per_bscan, 回傳 (min_bscan_q, n_bad_bscans)。"""
    import h5py
    mins, nbad = [], []
    for _, r in df.iterrows():
        mn, nb = np.nan, np.nan
        p = r.get("h5_path")
        try:
            with h5py.File(p, "r") as f:
                if "image_quality_per_bscan" in f:
                    q = np.asarray(f["image_quality_per_bscan"][:], dtype=float)
                    mn = float(np.nanmin(q))
                    nb = int(np.sum(q < min_quality))
        except Exception:
            pass
        mins.append(mn)
        nbad.append(nb)
    return mins, nbad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="stage0/index.parquet")
    ap.add_argument("--out", default="stage0/index_qc.parquet")
    ap.add_argument("--min-quality", type=float, default=15.0)
    ap.add_argument("--min-valid-ratio", type=float, default=0.9)
    ap.add_argument("--n-bscans", type=int, default=25)
    ap.add_argument("--check-bscans", action="store_true",
                    help="另讀 h5 的 image_quality_per_bscan 算 per-slice 品質 (慢)")
    args = ap.parse_args()

    df = pd.read_parquet(args.index)
    n = len(df)
    print(f"讀入 index: {args.index}  ({n} 列)")

    hard, soft = compute_qc(df, args.min_quality, args.min_valid_ratio, args.n_bscans)
    df["qc_flags"] = hard
    df["qc_soft"] = soft
    df["qc_pass"] = df["qc_flags"] == ""

    if args.check_bscans:
        print("讀 h5 計算 per-bscan 品質中 ...")
        mins, nbad = add_bscan_quality(df, args.min_quality)
        df["min_bscan_q"] = mins
        df["n_bad_bscans"] = nbad

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)

    # ---- 摘要 ----
    rules = ["read_error", "no_volume", "low_quality", "low_valid_ratio",
             "wrong_n_bscans", "width_inconsistent"]
    print(f"\n寫出: {args.out}")
    print("\n=== QC 摘要 ===")
    print("各硬性規則觸發數 (可重疊):")
    for rule in rules:
        c = df["qc_flags"].str.contains(rule).sum()
        if c:
            print(f"  {rule:18s}: {c}")
    n_pass = int(df["qc_pass"].sum())
    n_drop = n - n_pass
    print(f"\n通過 qc_pass: {n_pass}/{n} ({n_pass/n*100:.1f}%)   丟棄: {n_drop}")

    soft_n = (df["qc_soft"] != "").sum()
    if soft_n:
        print("軟性旗標 (不丟棄):")
        for rule in ["no_layers", "no_valid_mask"]:
            c = df["qc_soft"].str.contains(rule).sum()
            if c:
                print(f"  {rule:14s}: {c}")

    # patient / eye 維度
    pat_all = df["patient_id"].nunique()
    pat_pass = df.loc[df["qc_pass"], "patient_id"].nunique()
    print(f"\n病人數: 全部 {pat_all} -> 至少1顆通過 {pat_pass}")
    print("通過後眼別分布:")
    print(df.loc[df["qc_pass"], "laterality"].value_counts().to_string())

    # 列出被丟的檔 (方便目視)
    dropped = df.loc[~df["qc_pass"], ["h5_path", "image_quality",
                                      "valid_ascan_ratio", "vol_d", "qc_flags"]]
    if len(dropped):
        print(f"\n被丟棄的 {len(dropped)} 檔:")
        with pd.option_context("display.max_colwidth", 60):
            print(dropped.to_string(index=False))


if __name__ == "__main__":
    main()
