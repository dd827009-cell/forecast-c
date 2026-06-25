"""
Stage 0 - Module 5: 計算全資料強度 mean/std (供 dataloader 正規化)。

對每顆 qc_pass volume 跑「與正式訓練相同的前處理」(M2 強度轉換 + M3 幾何),
再用 valid_mask_256 排除無效 A-scan 後, 串流累計 sum / sumsq / count, 最後合併成
單通道 (灰階) 的 global mean/std。輸出 stage0/norm_stats.json。

為何在 256 後算: dataloader 餵給 backbone 的就是 256 volume, mean/std 必須在
同一個值域空間量測才有意義 (M2 已正規化到 [0,1], 這裡量的是該空間的分布)。

洩漏控制: 原則上 mean/std 應只用 train split。若提供 --manifest (M6 產出),
預設只在 split in --splits (預設 train) 的列上計算; 否則退化為對 index 的全部
qc_pass 計算 (pilot 方便, 會印警告)。

用法:
    # 先 M6 再 M5 (建議, 無洩漏):
    python stage0/m5_meanstd.py --manifest stage0/manifest.parquet --splits train
    # 或 pilot 直接對所有 qc_pass:
    python stage0/m5_meanstd.py --index stage0/index_qc.parquet
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

try:                                  # Windows 主控台預設 cp950, 統一改 utf-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_transform import transform_volume          # noqa: E402
from m3_geometry import process_volume              # noqa: E402
from _version import STAGE0_VERSION                 # noqa: E402


def accumulate_one(path):
    """讀單檔 -> M2+M3 -> 用 valid_mask_256 遮罩累計。

    回傳 (sum, sumsq, count, n_vox_total, err)。float64 避免溢位。
    """
    try:
        with h5py.File(path, "r") as f:
            vol = f["volume"][:].astype(np.float32)
            valid = f["valid_ascan_mask"][:] if "valid_ascan_mask" in f else None
            lat = f.attrs.get("laterality", "OD")
            lat = lat.decode() if isinstance(lat, bytes) else lat
        vol01 = transform_volume(vol, valid)
        res = process_volume(vol01, lat, valid_mask=valid)
        out = res["volume"].astype(np.float64)            # (D,256,256)
        vm256 = res["valid_mask_256"]                      # (D,256) or None
        if vm256 is not None:
            m = np.broadcast_to(vm256[:, None, :], out.shape)
            px = out[m]
        else:
            px = out.ravel()
        s = float(px.sum())
        ss = float((px * px).sum())
        c = int(px.size)
        return s, ss, c, int(out.size), None
    except Exception as e:
        return 0.0, 0.0, 0, 0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="stage0/index_qc.parquet",
                    help="M4 輸出 (含 qc_pass)。--manifest 未給時用此")
    ap.add_argument("--manifest", default=None,
                    help="M6 輸出 (含 split)。給了則只在 --splits 指定的列計算")
    ap.add_argument("--splits", default="train",
                    help="用逗號分隔的 split 名單 (僅在 --manifest 模式生效)")
    ap.add_argument("--out", default="stage0/norm_stats.json")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    if args.manifest and os.path.exists(args.manifest):
        df = pd.read_parquet(args.manifest)
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        df = df[df["split"].isin(splits)]
        scope = f"manifest split in {splits}"
        if "qc_pass" in df.columns:
            df = df[df["qc_pass"]]
    else:
        df = pd.read_parquet(args.index)
        df = df[df["qc_pass"]]
        scope = "index qc_pass (全部, 注意: 含 val/test, 僅 pilot 可接受)"
        print("[警告] 未提供 --manifest, 對所有 qc_pass 計算 mean/std (有輕微洩漏風險)。")

    paths = df["h5_path"].tolist()
    print(f"計算 mean/std 範圍: {scope}  共 {len(paths)} 顆 volume, workers={args.workers}")
    if not paths:
        print("無檔案, 結束。")
        return

    if args.workers <= 1:
        results = [accumulate_one(p) for p in tqdm(paths, desc="累計", unit="vol")]
    else:
        with Pool(args.workers) as pool:
            results = list(tqdm(pool.imap_unordered(accumulate_one, paths, chunksize=4),
                                total=len(paths), desc="累計", unit="vol"))

    tot_s = tot_ss = 0.0
    tot_c = tot_vox = 0
    errs = []
    for (s, ss, c, vox, err), p in zip(results, paths):
        if err:
            errs.append((p, err))
            continue
        tot_s += s; tot_ss += ss; tot_c += c; tot_vox += vox

    if tot_c == 0:
        print("沒有有效像素, 無法計算。錯誤:", errs[:5])
        return

    mean = tot_s / tot_c
    var = max(tot_ss / tot_c - mean * mean, 0.0)
    std = float(np.sqrt(var))
    valid_frac = tot_c / tot_vox if tot_vox else float("nan")

    stats = {
        "mean": mean,
        "std": std,
        "n_volumes": len(paths) - len(errs),
        "n_valid_pixels": tot_c,
        "n_total_pixels": tot_vox,
        "valid_pixel_fraction": valid_frac,
        "channels": 1,
        "masked_by": "valid_mask_256",
        "computed_on": scope,
        "transform_version": STAGE0_VERSION,
        "errors": errs,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)

    print(f"\n寫出: {args.out}")
    print(f"mean = {mean:.6f}   std = {std:.6f}")
    print(f"有效像素 {tot_c:,} / {tot_vox:,} ({valid_frac*100:.2f}%)  "
          f"volume {stats['n_volumes']}  錯誤 {len(errs)}")
    if errs:
        print("錯誤範例:", errs[:3])


if __name__ == "__main__":
    main()
