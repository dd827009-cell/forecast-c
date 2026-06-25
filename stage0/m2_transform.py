"""
Stage 0 - Module 2: 強度轉換 (raw 反射率 -> [0,1] 顯示值)。

配方沿用 scripts/make_report.py 第49-63行 (已驗證即生成現有 bscan PNG 的程式):
    sqrt(clip(vol,0)) -> 取非零值第15百分位當地板壓黑背景
    -> 對剩餘非零值取 global percentile[1, 99.9] -> clip 正規化到 [0,1]

升級點: 百分位統計用 valid_ascan_mask 排除邊緣無效 A-scan (影響<0.2%, 求嚴謹)。
特性: per-volume global (整個 volume 25 張切片共用同一組門檻 -> 跨切片一致, 符合 3D-MAE)。

只負責「轉換」, 不做翻轉/裁切/resize (那是 Module 3)。

用法 (驗證):
    python stage0/m2_transform.py
"""
import numpy as np

FLOOR_PCTL = 15.0    # 背景地板: 非零值的第15百分位
P_LO = 1.0           # 正規化下界百分位
P_HI = 99.9          # 正規化上界百分位


def transform_volume(vol, valid_mask=None):
    """raw volume (D,H,W) -> float32 [0,1] (D,H,W)。

    參數:
        vol: (D,H,W) 原始反射率 (float, 線性)。
        valid_mask: (D,W) bool, 每根 A-scan 是否有效; None 則用全部像素。
    回傳:
        (D,H,W) float32, 值域 [0,1]。
    其他回傳資訊透過 transform_volume_with_stats 取得。
    """
    out, _ = transform_volume_with_stats(vol, valid_mask)
    return out


def transform_volume_with_stats(vol, valid_mask=None):
    """同 transform_volume, 另回傳門檻統計 dict (供驗證/記錄)。"""
    disp = np.sqrt(np.clip(vol, 0, None)).astype(np.float32)
    D, H, W = disp.shape

    # 統計用的像素遮罩: 把 (D,W) 的 A-scan 有效性廣播到 (D,H,W)
    if valid_mask is not None:
        mask3d = np.broadcast_to(valid_mask[:, None, :], disp.shape)
    else:
        mask3d = np.ones_like(disp, dtype=bool)

    # ① 背景地板 (用有效且非零的像素算)
    stat = disp[mask3d]
    nz = stat[stat > 0]
    if nz.size == 0:
        return np.zeros_like(disp), {"floor": 0.0, "p_lo": 0.0, "p_hi": 1.0, "empty": True}
    floor = float(np.percentile(nz, FLOOR_PCTL))
    disp[disp < floor] = 0.0   # 套用到整個 volume (不只有效欄)

    # ② 正規化門檻 (地板後, 有效且非零的像素)
    stat2 = disp[mask3d]
    vv = stat2[stat2 > 0]
    if vv.size == 0:
        p_lo, p_hi = 0.0, 1.0
    else:
        p_lo = float(np.percentile(vv, P_LO))
        p_hi = float(np.percentile(vv, P_HI))
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0

    # ③ 正規化到 [0,1]
    disp = np.clip((disp - p_lo) / (p_hi - p_lo), 0.0, 1.0).astype(np.float32)
    stats = {"floor": floor, "p_lo": p_lo, "p_hi": p_hi,
             "out_mean": float(disp.mean()), "empty": False}
    return disp, stats


# ============================ 驗證 ============================
def _verify():
    import sys, io, glob
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    import h5py
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 從 index 挑跨品質的樣本: 最低/中位/最高 quality
    df = pd.read_parquet("stage0/index.parquet")
    df = df[df["n_bscans"] == 25].sort_values("image_quality")
    picks = [df.iloc[0], df.iloc[len(df) // 2], df.iloc[-1]]

    fig, axes = plt.subplots(2, len(picks), figsize=(6 * len(picks), 9))
    for j, row in enumerate(picks):
        p = row["h5_path"]
        with h5py.File(p, "r") as f:
            vol = f["volume"][:].astype(np.float32)
            valid = f["valid_ascan_mask"][:]
        q = row["image_quality"]

        out_m, st_m = transform_volume_with_stats(vol, valid)       # 用 mask
        out_n, st_n = transform_volume_with_stats(vol, None)        # 不用 mask
        mid = vol.shape[0] // 2

        print(f"\n[{j}] q={q:.1f}  {p.split(chr(92))[-1]}")
        print(f"    用mask : floor={st_m['floor']:.3f} p_lo={st_m['p_lo']:.3f} "
              f"p_hi={st_m['p_hi']:.3f} out_mean={st_m['out_mean']:.3f}")
        print(f"    不用mask: floor={st_n['floor']:.3f} p_lo={st_n['p_lo']:.3f} "
              f"p_hi={st_n['p_hi']:.3f} out_mean={st_n['out_mean']:.3f}")
        # 跨切片一致性檢查: 各切片轉換後的 mean (應隨解剖變化, 但用同一尺度)
        per_slice_mean = [out_m[d].mean() for d in range(vol.shape[0])]
        print(f"    跨切片 out_mean: min={min(per_slice_mean):.3f} "
              f"max={max(per_slice_mean):.3f} (共用同一組 floor/p_lo/p_hi)")

        axes[0, j].imshow(out_m[mid], cmap="gray", vmin=0, vmax=1, aspect="auto")
        axes[0, j].set_title(f"q={q:.1f}  mid B-scan (transformed)"); axes[0, j].axis("off")
        # 差異圖: 用mask vs 不用mask
        diff = np.abs(out_m[mid] - out_n[mid])
        im = axes[1, j].imshow(diff, cmap="hot", aspect="auto", vmin=0, vmax=0.05)
        axes[1, j].set_title(f"|mask - no_mask|  (max={diff.max():.3f})"); axes[1, j].axis("off")
    fig.colorbar(im, ax=axes[1, :], fraction=0.02)
    fig.suptitle("Module 2 強度轉換驗證: 上=轉換結果, 下=用/不用valid_mask差異", fontsize=13)
    out_png = "stage0/m2_verify.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"\n驗證圖已存: {out_png}")


if __name__ == "__main__":
    _verify()
