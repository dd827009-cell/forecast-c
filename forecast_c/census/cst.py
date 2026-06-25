"""中央亞區厚度 CST（Central Subfield Thickness）— A-1 普查 + baseline 嚴重度共用。

從 M7b per-eye 層分割（ilm / rpe_bm，native 列空間 px）算:
  - 厚度圖 µm = (rpe - ilm) * scale_axial_um_per_px；NaN（分割失敗）與層交叉(<0) → 無效。
  - 中央 1mm CST = en-face 平面（切片軸 × 橫向軸）距中心 ≤0.5mm 圈內有效厚度平均。
  - 正規化: 對 **train split** 的 CST 算 mean/std → z-score（baseline_severity 條件用）。

純 numpy / stdlib（無 torch / cv2 / h5py）→ 任何機器可跑。self-test 見檔尾。
"""
import json

import numpy as np

# 軸向 µm/px（同 stage0 m3_geometry）。lateral/bscan 若 meta 缺則用黃斑 6mm-FOV 推估。
DEFAULT_AXIAL_UM_PER_PX = 3.87167
DEFAULT_LATERAL_MM_PER_PX = 6.0 / 512.0      # 6mm 橫向 FOV / 512 欄
DEFAULT_BSCAN_SPACING_MM = 6.0 / 25.0        # 6mm 慢軸 / 25 B-scan


def thickness_um(ilm, rpe, axial_um_per_px=DEFAULT_AXIAL_UM_PER_PX, valid=None):
    """層座標 → 厚度圖（µm）。NaN 與非物理(<0, 層交叉) → NaN。

    ilm, rpe: (T, W) float，native 列空間（px）; valid: (T,W) bool 可選額外有效遮罩。
    """
    ilm = np.asarray(ilm, np.float32)
    rpe = np.asarray(rpe, np.float32)
    th = (rpe - ilm) * float(axial_um_per_px)
    bad = ~np.isfinite(th)
    if valid is not None:
        bad |= ~np.asarray(valid, bool)
    th = th.copy()
    th[bad] = np.nan
    th[th < 0] = np.nan                       # 層交叉 → 無效（物理不可能）
    return th


def central_subfield_thickness(ilm, rpe, *,
                               axial_um_per_px=DEFAULT_AXIAL_UM_PER_PX,
                               lateral_mm_per_px=DEFAULT_LATERAL_MM_PER_PX,
                               bscan_spacing_mm=DEFAULT_BSCAN_SPACING_MM,
                               diameter_mm=1.0, ilm_valid=None, rpe_valid=None,
                               center_rc=None):
    """中央亞區厚度 CST（µm）+ 取樣到的有效點數。

    回傳 (cst_um: float, n_used: int)。圈內無有效厚度時 cst_um=NaN。
    """
    valid = None
    if ilm_valid is not None or rpe_valid is not None:
        iv = np.ones_like(ilm, bool) if ilm_valid is None else np.asarray(ilm_valid, bool)
        rv = np.ones_like(rpe, bool) if rpe_valid is None else np.asarray(rpe_valid, bool)
        valid = iv & rv
    th = thickness_um(ilm, rpe, axial_um_per_px, valid)        # (T,W) µm, NaN=無效
    T, W = th.shape

    # 缺 scale 時退回 FOV 推估值（並保證為正）
    lat = float(lateral_mm_per_px) if lateral_mm_per_px and np.isfinite(lateral_mm_per_px) else DEFAULT_LATERAL_MM_PER_PX
    bsp = float(bscan_spacing_mm) if bscan_spacing_mm and np.isfinite(bscan_spacing_mm) else DEFAULT_BSCAN_SPACING_MM

    cr, cc = ((T - 1) / 2.0, (W - 1) / 2.0) if center_rc is None else center_rc
    row_mm = (np.arange(T) - cr) * bsp                        # 沿 B-scan 慢軸 mm
    col_mm = (np.arange(W) - cc) * lat                        # 沿橫向 mm
    rr, ccc = np.meshgrid(row_mm, col_mm, indexing="ij")
    inside = (rr ** 2 + ccc ** 2) <= (diameter_mm / 2.0) ** 2

    sel = inside & np.isfinite(th)
    n_used = int(sel.sum())
    if n_used == 0:
        return float("nan"), 0
    return float(np.mean(th[sel])), n_used


def _meta_scales(meta):
    """從 M7b meta dict 取三個 scale（缺則 None → 由下游用預設）。"""
    def g(k):
        v = meta.get(k)
        try:
            v = float(v)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None
    return g("scale_axial_um_per_px"), g("scale_lateral_mm_per_px"), g("scale_bscan_spacing_mm")


def cst_from_npz(path, diameter_mm=1.0, center_rc=None):
    """讀 M7b per-eye npz → CST（µm）。回傳 (key, cst_um, n_used)。

    npz 內: ilm/rpe/ilm_valid/rpe_valid + meta（0-d object array 夾帶 JSON 字串）。
    """
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    ax, lat, bsp = _meta_scales(meta)
    cst, n = central_subfield_thickness(
        d["ilm"], d["rpe"],
        axial_um_per_px=ax or DEFAULT_AXIAL_UM_PER_PX,
        lateral_mm_per_px=lat or DEFAULT_LATERAL_MM_PER_PX,
        bscan_spacing_mm=bsp or DEFAULT_BSCAN_SPACING_MM,
        diameter_mm=diameter_mm,
        ilm_valid=d.get("ilm_valid"), rpe_valid=d.get("rpe_valid"),
        center_rc=center_rc)
    return meta.get("key"), cst, n


def cst_stats(cst_values):
    """對一組 CST（µm，可含 NaN）算 mean/std，供 normalize_cst。應只用 **train split**。"""
    arr = np.asarray([c for c in cst_values if c is not None and np.isfinite(c)], np.float64)
    if arr.size == 0:
        raise ValueError("沒有有效 CST 值可統計")
    return {"mean": float(arr.mean()), "std": float(arr.std() + 1e-6), "n": int(arr.size)}


def normalize_cst(cst_um, ref_mean, ref_std):
    """CST（µm）→ z-score 純量（餵 predictor cond 的 baseline_severity）。
    NaN（無有效層）→ 0.0（= 平均水準，不偏置條件）。"""
    if cst_um is None or not np.isfinite(cst_um):
        return 0.0
    return float((cst_um - ref_mean) / (ref_std + 1e-6))


# --------------------------------------------------------------------------- #
# 自我驗證: 合成已知厚度的層 → 驗 CST 算對（不需真實資料 / torch）
# 跑: python forecast_c/census/cst.py
# --------------------------------------------------------------------------- #
def _self_test():
    print("[self-test] cst: 合成層 → CST µm / 正規化 ...")
    T, W = 25, 512
    ax = DEFAULT_AXIAL_UM_PER_PX

    # 1) 均勻厚度 rpe-ilm 恆 80px → CST 應 = 80*ax µm，與圈大小無關
    ilm = np.full((T, W), 100.0, np.float32)
    rpe = ilm + 80.0
    cst, n = central_subfield_thickness(ilm, rpe, axial_um_per_px=ax)
    assert abs(cst - 80.0 * ax) < 1e-3, (cst, 80.0 * ax)
    assert n > 0
    print(f"  [OK] 均勻 80px → CST={cst:.2f}µm, 圈內 {n} 點")

    # 2) 中央凸起 → 中央 1mm CST 高於全域平均
    cr, cc = (T - 1) / 2.0, (W - 1) / 2.0
    rr, ccc = np.meshgrid(np.arange(T) - cr, np.arange(W) - cc, indexing="ij")
    bump = 80.0 + 40.0 * np.exp(-((rr / 3.0) ** 2 + (ccc / 60.0) ** 2))
    rpe2 = ilm + bump.astype(np.float32)
    cst2, _ = central_subfield_thickness(ilm, rpe2, axial_um_per_px=ax, diameter_mm=1.0)
    global_mean = np.mean((rpe2 - ilm)) * ax
    assert cst2 > global_mean and cst2 > 100.0 * ax, (cst2, global_mean)
    print(f"  [OK] 中央凸起 → 中央1mm CST={cst2:.2f}µm > 全域平均 {global_mean:.2f}µm")

    # 3) NaN / 層交叉 視為無效，不污染平均
    ilm3, rpe3 = ilm.copy(), rpe.copy()
    rpe3[int(cr), :] = np.nan
    rpe3[int(cr) + 1, int(cc)] = ilm3[int(cr) + 1, int(cc)] - 50.0    # 層交叉(<0)
    cst3, n3 = central_subfield_thickness(ilm3, rpe3, axial_um_per_px=ax)
    assert np.isfinite(cst3) and abs(cst3 - 80.0 * ax) < 1e-3 and n3 < n
    print(f"  [OK] NaN+層交叉排除 → CST 仍={cst3:.2f}µm, 有效點 {n3}<{n}")

    # 4) 圈內全無效 → NaN
    cst4, n4 = central_subfield_thickness(np.full((T, W), np.nan, np.float32),
                                          np.full((T, W), np.nan, np.float32))
    assert not np.isfinite(cst4) and n4 == 0
    print("  [OK] 全無效 → CST=NaN, n=0")

    # 5) 統計 + 正規化
    vals = [80.0 * ax, 100.0 * ax, 120.0 * ax, float("nan"), None]
    st = cst_stats(vals)
    assert st["n"] == 3
    z_hi = normalize_cst(120.0 * ax, st["mean"], st["std"])
    z_lo = normalize_cst(80.0 * ax, st["mean"], st["std"])
    assert z_hi > 0 > z_lo and normalize_cst(float("nan"), st["mean"], st["std"]) == 0.0
    print(f"  [OK] cst_stats n={st['n']}; z(120px)={z_hi:.2f}>0>z(80px)={z_lo:.2f}; NaN→0")

    # 6) scale 缺失退回預設（不炸）
    cst6, _ = central_subfield_thickness(ilm, rpe, lateral_mm_per_px=None,
                                         bscan_spacing_mm=float("nan"))
    assert np.isfinite(cst6)
    print("  [OK] scale 缺失 → 退回 FOV 推估，仍可算")
    print("[self-test OK]")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _self_test()
