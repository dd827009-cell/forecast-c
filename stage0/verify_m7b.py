"""讀回 M7b 抽存的 .npz 驗證內容 (shape/dtype/NaN/值域/層不交叉)。
用法: python stage0/verify_m7b.py [assets_dir]   (預設 assets)
"""
import sys, os, glob, json
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "assets"
mani = os.path.join(d, "manifest_assets.parquet")
npzs = sorted(glob.glob(os.path.join(d, "*", "*.npz")))
print(f"assets 目錄: {d}")
print(f"manifest_assets.parquet 存在: {os.path.exists(mani)}")
print(f"產出 .npz 檔數: {len(npzs)}")
if not npzs:
    sys.exit("沒有 .npz")

# 各 split 計數
from collections import Counter
c = Counter(os.path.basename(os.path.dirname(p)) for p in npzs)
print("各 split:", dict(c))

# 抽第一顆細看
p = npzs[0]
z = np.load(p)
print(f"\n抽樣: {os.path.relpath(p, d)}")
print(f"  keys: {list(z.keys())}")
EXPECT = {
    "ilm": ((25, 512), "float32"), "rpe": ((25, 512), "float32"),
    "ilm_valid": ((25, 512), "bool"), "rpe_valid": ((25, 512), "bool"),
    "valid_mask_512": ((25, 512), "bool"),
    "ascan_pos_ir": ((25, 512, 2), "float32"), "ir": ((768, 768), "uint8"),
}
ok = True
for k, (shp, dt) in EXPECT.items():
    if k not in z:
        print(f"  [✗] 缺 {k}"); ok = False; continue
    a = z[k]
    good = (a.shape == shp and str(a.dtype) == dt)
    ok &= good
    extra = ""
    if k in ("ilm", "rpe"):
        fin = np.isfinite(a)
        extra = f" finite={fin.mean()*100:.1f}% 範圍[{np.nanmin(a):.1f},{np.nanmax(a):.1f}]"
    elif k.endswith("valid") or k == "valid_mask_512":
        extra = f" True={a.mean()*100:.1f}%"
    elif k == "ir":
        extra = f" 範圍[{a.min()},{a.max()}]"
    elif k == "ascan_pos_ir":
        extra = f" 範圍[{np.nanmin(a):.0f},{np.nanmax(a):.0f}]"
    print(f"  [{'✓' if good else '✗'}] {k:16s} shape={str(a.shape):14s} dtype={a.dtype}{extra}")

# ILM 應在 RPE 上方 (層不交叉): ilm < rpe 在兩者皆有效處
if "ilm" in z and "rpe" in z:
    both = np.isfinite(z["ilm"]) & np.isfinite(z["rpe"])
    if both.any():
        frac = float((z["ilm"][both] <= z["rpe"][both]).mean())
        print(f"\n  ILM<=RPE (層不交叉) 在有效處比例 = {frac*100:.1f}%")
print("\n" + ("[readback OK]" if ok else "[readback 有問題]"))
