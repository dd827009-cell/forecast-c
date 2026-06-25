"""
Stage 0 - Module 1: 掃描所有 .h5, 讀 attrs 與 volume shape, 建立索引表。
這是後續 QC / patient-level split / 統計的基礎。

用法:
    python stage0/m1_build_index.py --root h5_output --out stage0/index.parquet
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import h5py
import pandas as pd
from tqdm import tqdm

try:                                  # Windows 主控台預設 cp950, 統一改 utf-8 避免印符號報錯
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 需要從 attrs 抓的欄位 (缺的填 None)
ATTR_KEYS = [
    "patient_id", "laterality", "visit_id", "longitudinal_key",
    "image_quality", "valid_ascan_ratio", "n_bscans",
    "age_at_visit_years", "sex", "acquisition_time_utc",
    "bscan_height", "bscan_width", "segmentation_types_available",
    "source_sdb_path",
    # 物理尺度 (A1): 下游 FOV-crop 物理一致性 與 thickness µm 換算都依賴這些為「全域常數」。
    # 必須實際抓進來才能驗證假設, 並讓 manifest 帶 per-volume scale (萬一非常數也能正確換算)。
    "scale_axial_um_per_px",       # 軸向 (深度) µm/px, 預期 ~3.87167
    "scale_lateral_mm_per_px",     # 橫向 mm/px, 預期 0.01125 (=11.25 µm/px)
    "scale_bscan_spacing_mm",      # 慢軸 B-scan 間距 mm, 預期 0.24
    "bscan_spacing_deg_per_index",
    "fovea_ir_x", "fovea_ir_y",    # 黃斑中心在 IR 的座標 (未來黃斑置中 / COEP), 抓進來免費
]

# 預期為「全域常數」的物理尺度欄位 (A1 驗證對象)
SCALE_KEYS = ["scale_axial_um_per_px", "scale_lateral_mm_per_px",
              "scale_bscan_spacing_mm", "bscan_spacing_deg_per_index"]


def _decode(v):
    return v.decode() if isinstance(v, bytes) else v


def scan_one(path):
    """讀單檔, 回傳一個 dict; 失敗回傳含 error 的 dict。"""
    rec = {"h5_path": path}
    try:
        with h5py.File(path, "r") as f:
            a = f.attrs
            for k in ATTR_KEYS:
                rec[k] = _decode(a[k]) if k in a else None
            # volume shape (不載入資料, 只讀 shape)
            if "volume" in f:
                d, h, w = f["volume"].shape
                rec["vol_d"], rec["vol_h"], rec["vol_w"] = int(d), int(h), int(w)
            else:
                rec["vol_d"] = rec["vol_h"] = rec["vol_w"] = None
            rec["has_ir"] = "ir" in f
            rec["has_ilm"] = "ilm_y" in f
            rec["has_rpe_bm"] = "rpe_bm_y" in f
            rec["has_valid_mask"] = "valid_ascan_mask" in f

            # M3 對齊前提: 附帶陣列的 W 軸 (axis=1) 必須 == volume 寬度。
            # 否則 M3 對 volume 與這些陣列套同一個 crop 會錯位 -> 此處先自動標記。
            vw = rec["vol_w"]
            aux_w = {}
            for k in ("ilm_y", "rpe_bm_y", "valid_ascan_mask", "ascan_pos_ir"):
                aux_w[k] = int(f[k].shape[1]) if k in f else None
            present_w = [w for w in aux_w.values() if w is not None]
            if vw is None or not present_w:
                rec["w_consistent"] = None
                rec["aux_w_detail"] = None
            else:
                ok = all(w == vw for w in present_w)
                rec["w_consistent"] = ok
                rec["aux_w_detail"] = None if ok else \
                    "vol_w=%s; " % vw + "; ".join(f"{k}={w}" for k, w in aux_w.items())
        rec["error"] = None
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="h5_output", help="放 .h5 的根目錄")
    ap.add_argument("--out", default="stage0/index.parquet")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="平行進程數 (預設 = CPU 核心數 - 1; 設 1 則序列執行)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "**", "*.h5"), recursive=True))
    print(f"找到 {len(files)} 個 .h5,  workers={args.workers}")
    if not files:
        print("沒有檔案, 結束。")
        return

    if args.workers <= 1:
        records = [scan_one(p) for p in tqdm(files, desc="掃描 .h5", unit="file")]
    else:
        with Pool(args.workers) as pool:
            records = list(tqdm(pool.imap_unordered(scan_one, files, chunksize=16),
                                total=len(files), desc="掃描 .h5", unit="file"))
    # imap_unordered 會打亂順序, 依 h5_path 排序還原穩定輸出
    df = pd.DataFrame(records).sort_values("h5_path").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)

    # ---- 摘要 ----
    n_err = df["error"].notna().sum()
    print(f"\n寫出索引: {args.out}  ({len(df)} 列, {n_err} 個讀取錯誤)")
    print("\n=== 摘要 ===")
    print("唯一病人數:", df["patient_id"].nunique())
    print("眼別分布:\n", df["laterality"].value_counts().to_string())
    print("\nvolume 寬度分布:\n", df["vol_w"].value_counts().to_string())
    print("\nn_bscans 分布:\n", df["n_bscans"].value_counts().to_string())
    print("\nimage_quality: min=%.1f mean=%.1f max=%.1f" % (
        df["image_quality"].min(), df["image_quality"].mean(), df["image_quality"].max()))
    print("每病人 volume 數: min=%d mean=%.1f max=%d" % (
        df.groupby("patient_id").size().min(),
        df.groupby("patient_id").size().mean(),
        df.groupby("patient_id").size().max()))

    # ---- 物理尺度全域常數驗證 (A1) ----
    # 下游 512px FOV 物理一致性、thickness µm = diff × axial 都假設這些是全資料常數。
    # 此處實測: 印出每個 scale 的唯一值, 若非單一值就明確警告 (下游需改用 per-volume scale)。
    # 用相對容差判定 (容忍 float32 表示誤差; 真正物理差異會遠大於此)。
    SCALE_RTOL = 1e-4
    print("\n=== 物理尺度全域常數驗證 (A1, 相對容差 %.0e) ===" % SCALE_RTOL)
    all_constant = True
    for k in SCALE_KEYS:
        if k not in df.columns:
            print(f"  {k:28s}: (欄位不存在)"); continue
        s = df[k].dropna().astype(float)
        if s.empty:
            print(f"  {k:28s}: (全空)"); continue
        vmin, vmax, vmean = s.min(), s.max(), s.mean()
        rel_range = (vmax - vmin) / abs(vmean) if vmean else 0.0
        n_exact = s.nunique()
        if rel_range <= SCALE_RTOL:
            note = "  (float32 表示誤差, 物理上同值)" if n_exact > 1 else ""
            print(f"  {k:28s}: 常數 ≈ {vmean:.6g}  [OK]{note}")
        else:
            all_constant = False
            print(f"  {k:28s}: [警告] 非常數! {n_exact} 種值 範圍[{vmin:.6g},{vmax:.6g}] "
                  f"相對range={rel_range:.2e} -> 下游 µm 換算須改用 per-volume scale")
    if all_constant:
        print("  結論: 所有尺度在容差內全域常數, doc 物理一致性假設成立 [OK]")
    else:
        print("  結論: [警告] 有尺度非全域常數! 需在 M3/manifest 改用 per-volume scale 做 µm 換算")

    # 附帶陣列 W 一致性 (M3 對齊前提)
    if "w_consistent" in df:
        n_bad_w = (df["w_consistent"] == False).sum()  # noqa: E712 (排除 None)
        print("\n附帶陣列寬度一致性: %d 檔不一致 (需排除, 會破壞 M3 對齊)" % n_bad_w)
        if n_bad_w:
            print(df.loc[df["w_consistent"] == False,  # noqa: E712
                         ["h5_path", "aux_w_detail"]].head(20).to_string())
    if n_err:
        print("\n讀取錯誤範例:\n", df[df["error"].notna()][["h5_path", "error"]].head().to_string())


if __name__ == "__main__":
    main()
