"""OCT h5 讀取器 — 把 h5_output 的 study 讀成模型/教師/厚度頭要的張量。

h5 schema（每 study 一檔，stage0 產出）:
  volume (25,496,512) f32 | ilm_y/rpe_bm_y (25,512) | valid_ascan_mask (25,512) bool
  ascan_pos_ir (25,512,2) | ir (768,768) u8 | attrs: longitudinal_key=patient_id::OD/OS,
  acquisition_time_utc, scale_axial_um_per_px, age_at_visit_years, sex, laterality...

提供:
  read_h5            : → dict(volume, ilm, rpe, valid, attrs)
  octcube_volume     : OCT cube (25,496,512) → 餵 OCTCubeTokenEncoder.prep
  bscans_for_teachers: (25,1,496,512) 每張 B-scan → 教師特徵
  thickness_gt       : (25,512) 厚度圖 µm + valid mask（厚度頭真 GT；BM−ILM）
  cst                : 中央 1mm CST（純量；census/baseline）
"""
import h5py
import numpy as np
import torch

from forecast_c.census.cst import (thickness_um, central_subfield_thickness,
                                    DEFAULT_AXIAL_UM_PER_PX)


def read_h5(path):
    """讀一個 study h5 → dict。"""
    with h5py.File(path, "r") as h:
        d = {"volume": h["volume"][:], "ilm": h["ilm_y"][:], "rpe": h["rpe_bm_y"][:],
             "valid": h["valid_ascan_mask"][:], "attrs": dict(h.attrs)}
    return d


def _axial(attrs):
    v = attrs.get("scale_axial_um_per_px", DEFAULT_AXIAL_UM_PER_PX)
    try:
        v = float(v)
        return v if np.isfinite(v) and v > 0 else DEFAULT_AXIAL_UM_PER_PX
    except (TypeError, ValueError):
        return DEFAULT_AXIAL_UM_PER_PX


def octcube_volume(path):
    """OCT cube (25,496,512) f32（餵 OCTCubeTokenEncoder.prep）。"""
    return read_h5(path)["volume"]


def bscans_for_teachers(path):
    """每張 B-scan → (n_bscan, 1, H, W) tensor（教師 extractor 吃；內部會 resize/3ch/norm）。"""
    vol = read_h5(path)["volume"]                              # (25,496,512)
    return torch.from_numpy(vol).float().unsqueeze(1)         # (25,1,496,512)


def thickness_gt(path):
    """厚度頭真 GT: (n_bscan, W) 厚度圖 µm = (rpe−ilm)·axial + valid mask。

    回傳 (thickness (25,512) f32 NaN→0, mask (25,512) bool)。NaN/層交叉 → mask=False。
    """
    d = read_h5(path)
    th = thickness_um(d["ilm"], d["rpe"], _axial(d["attrs"]), valid=d["valid"])  # NaN=無效
    mask = np.isfinite(th)
    th = np.nan_to_num(th, nan=0.0)
    return torch.from_numpy(th.astype(np.float32)), torch.from_numpy(mask)


def cst(path, diameter_mm=1.0):
    """中央 1mm CST（µm）。"""
    d = read_h5(path)
    a = d["attrs"]
    val, _ = central_subfield_thickness(
        d["ilm"], d["rpe"], axial_um_per_px=_axial(a),
        lateral_mm_per_px=float(a.get("scale_lateral_mm_per_px", 0) or 0) or None,
        bscan_spacing_mm=float(a.get("scale_bscan_spacing_mm", 0) or 0) or None,
        diameter_mm=diameter_mm, ilm_valid=d["valid"], rpe_valid=d["valid"])
    return val


# ───────────────────────── self-test（需 h5_output）`python -m forecast_c.data.oct_h5` ─────────────────────────
if __name__ == "__main__":
    import glob, sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    fs = sorted(glob.glob("h5_output/**/*.h5", recursive=True))
    assert fs, "找不到 h5_output/*.h5"
    f = fs[0]
    d = read_h5(f)
    print("study:", f.split("/")[-1], "| longitudinal_key:", d["attrs"]["longitudinal_key"])
    th, mask = thickness_gt(f)
    print(f"  厚度 GT: {tuple(th.shape)} 有效 {int(mask.sum())}/{mask.numel()}, "
          f"有效厚度範圍 [{float(th[mask].min()):.0f}, {float(th[mask].max()):.0f}] µm")
    print(f"  中央 CST: {cst(f):.1f} µm")
    bs = bscans_for_teachers(f)
    print(f"  B-scans: {tuple(bs.shape)}")
    assert th.shape == (25, 512) and mask.shape == (25, 512) and bs.shape == (25, 1, 496, 512)
    assert mask.sum() > 0 and 100 < float(th[mask].mean()) < 600    # 視網膜厚度合理範圍
    print("oct_h5 self-test 通過 ✅（真 h5 → 厚度 GT / CST / B-scan）")
