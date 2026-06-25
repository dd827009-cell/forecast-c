"""獨立驗證 M5 (norm_stats) 與 M6 (manifest) 是否正確。
刻意用與 m5/m6 不同的程式路徑重算 / 交叉比對, 不靠原程式自證。

用法: python stage0/verify_m5_m6.py
"""
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_transform import transform_volume
from m3_geometry import process_volume
from _version import STAGE0_VERSION

# 可移植: 從本檔位置推導 repo 根 (本機/容器/L40 皆適用); 可用 PRETRAIN_ROOT 覆蓋。
ROOT = os.environ.get("PRETRAIN_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_QC = os.path.join(ROOT, "stage0/index_qc.parquet")
MANIFEST = os.path.join(ROOT, "stage0/manifest.parquet")
NORM = os.path.join(ROOT, "stage0/norm_stats.json")

PASS = "✓"
FAIL = "✗"
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  -> {detail}" if detail else ""))


print("=" * 70)
print("M6 manifest 驗證")
print("=" * 70)
idx = pd.read_parquet(INDEX_QC)
man = pd.read_parquet(MANIFEST)

# (1) 列數與 index 對齊
check("manifest 列數 == index 列數", len(man) == len(idx), f"{len(man)} vs {len(idx)}")

# (2) 必要欄位齊全
req = ["patient_id", "eye", "visit_id", "longitudinal_key", "shard_path",
       "H_orig", "W_orig", "image_quality", "valid_ascan_ratio", "age",
       "sex", "split", "transform_version", "qc_flags", "qc_pass"]
miss = [c for c in req if c not in man.columns]
check("必要欄位齊全", not miss, f"缺: {miss}" if miss else "全部存在")

# (3) transform_version 正確
check("transform_version 一致", (man["transform_version"] == STAGE0_VERSION).all(),
      STAGE0_VERSION)

# (4) dropped 列 <=> qc_pass==False (M4 結果)
dropped_set = set(man.loc[man["split"] == "dropped", "h5_path"])
failqc_set = set(idx.loc[~idx["qc_pass"], "h5_path"])
check("dropped 集合 == qc_pass==False 集合",
      dropped_set == failqc_set, f"dropped={len(dropped_set)} failqc={len(failqc_set)}")

# (5) 通過 QC 的 volume 全部有非 dropped split
passed = man[man["qc_pass"]]
check("所有 qc_pass volume 都被指派 train/val/test",
      passed["split"].isin(["train", "val", "test"]).all())

# (6) ★ 病人級洩漏: 每位病人的所有 (非dropped) volume 必須同一 split
grp = man[man["split"] != "dropped"].groupby("patient_id")["split"].nunique()
leak = grp[grp > 1]
check("無病人跨 split 洩漏 (鐵則)", len(leak) == 0,
      "OK" if len(leak) == 0 else f"洩漏病人: {list(leak.index)}")

# (7) ★ 縱向/雙眼也算病人級: 同病人不同 eye/visit 不可分散
#     用 longitudinal_key 再驗一次 (longitudinal_key 內含 patient::eye)
man2 = man[man["split"] != "dropped"].copy()
man2["pid_from_key"] = man2["longitudinal_key"].astype(str).str.split("::").str[0]
# pid_from_key 與 patient_id 應一致 (資料完整性)
consistent = (man2["pid_from_key"] == man2["patient_id"].astype(str)).mean()
check("longitudinal_key 前綴 == patient_id", consistent > 0.99,
      f"{consistent*100:.1f}% 一致")

# (8) split 比例合理 (volume 級), train 應 ~96%
vc = man[man["qc_pass"]]["split"].value_counts()
n_pass = len(passed)
train_frac = vc.get("train", 0) / n_pass
check("train 佔 qc_pass 比例接近 0.96 (小樣本容差)", 0.85 <= train_frac <= 1.0,
      f"train={vc.get('train',0)}/{n_pass}={train_frac*100:.1f}%, "
      f"val={vc.get('val',0)} test={vc.get('test',0)}")

# (9) 病人級互斥計數: train+val+test 病人數 == 總通過病人數
pat_per_split = passed.groupby("split")["patient_id"].apply(set)
all_pat = set().union(*pat_per_split.values)
sum_pat = sum(len(s) for s in pat_per_split.values)
check("各 split 病人集合互斥且涵蓋全部",
      sum_pat == len(all_pat) == passed["patient_id"].nunique(),
      f"sum={sum_pat} union={len(all_pat)} total={passed['patient_id'].nunique()}")

# (10) A2 縱向覆蓋: eye_n_visits 欄存在且與獨立重算一致; val/test 有 progression pair
check("A2: manifest 含 eye_n_visits 欄", "eye_n_visits" in man.columns)
mp = man[man["qc_pass"]]
ev_indep = mp.groupby("longitudinal_key")["visit_id"].nunique()
# eye_n_visits 應等於該 longitudinal_key 的 qc_pass visit 數 (獨立重算)
merged = mp.assign(_ev=mp["longitudinal_key"].map(ev_indep))
check("A2: eye_n_visits == 獨立重算的同眼 visit 數",
      (merged["eye_n_visits"].astype("Int64") == merged["_ev"].astype("Int64")).all())
for s in ["val", "test"]:
    sub = mp[mp["split"] == s]
    pair_eyes = int(sub.groupby("longitudinal_key")["visit_id"].nunique().ge(2).sum())
    check(f"A2: {s} 至少 1 隻 progression pair 眼 (同眼>=2visit)", pair_eyes >= 1,
          f"{s} pair_eyes={pair_eyes}")

print("\n" + "=" * 70)
print("M5 norm_stats 驗證 (獨立重算交叉比對)")
print("=" * 70)
with open(NORM, encoding="utf-8") as f:
    stats = json.load(f)
print(f"  norm_stats.json: mean={stats['mean']:.6f} std={stats['std']:.6f} "
      f"n_vol={stats['n_volumes']} valid={stats['valid_pixel_fraction']*100:.2f}%")

# 取 manifest 的 train 清單 (與 m5 同範圍)
train_paths = man.loc[(man["split"] == "train") & (man["qc_pass"]), "h5_path"].tolist()
check("M5 計算範圍 == manifest train 顆數",
      stats["n_volumes"] == len(train_paths), f"{stats['n_volumes']} vs {len(train_paths)}")
check("M5 computed_on 標註 train", "train" in stats["computed_on"], stats["computed_on"])
check("M5 transform_version 一致", stats["transform_version"] == STAGE0_VERSION)


def per_volume_stats(path):
    """獨立路徑: 回傳該 volume 有效像素的 (sum, sumsq, count)。"""
    with h5py.File(path, "r") as f:
        vol = f["volume"][:].astype(np.float32)
        valid = f["valid_ascan_mask"][:] if "valid_ascan_mask" in f else None
        lat = f.attrs.get("laterality", "OD")
        lat = lat.decode() if isinstance(lat, bytes) else lat
    v01 = transform_volume(vol, valid)
    res = process_volume(v01, lat, valid_mask=valid)
    out = res["volume"].astype(np.float64)
    vm = res["valid_mask_256"]
    px = out[np.broadcast_to(vm[:, None, :], out.shape)] if vm is not None else out.ravel()
    return px.sum(), (px * px).sum(), px.size, out.size


# 全量獨立重算 (two-pass 概念: 先全收 sum/sumsq, 再算; 與 m5 的單pass累計互相佐證)
S = SS = 0.0
C = VT = 0
for p in train_paths:
    s, ss, c, vt = per_volume_stats(p)
    S += s; SS += ss; C += c; VT += vt
mean2 = S / C
std2 = float(np.sqrt(max(SS / C - mean2 * mean2, 0.0)))
check("獨立重算 mean 與 M5 相符 (<1e-6)", abs(mean2 - stats["mean"]) < 1e-6,
      f"重算 {mean2:.8f} vs M5 {stats['mean']:.8f}")
check("獨立重算 std 與 M5 相符 (<1e-6)", abs(std2 - stats["std"]) < 1e-6,
      f"重算 {std2:.8f} vs M5 {stats['std']:.8f}")
check("有效像素數一致", C == stats["n_valid_pixels"], f"{C} vs {stats['n_valid_pixels']}")

# 第三條獨立路徑: 抽 1 顆 volume, 用 np.mean/np.std 直接算 (不經累計), 比對其貢獻合理
sample = train_paths[0]
with h5py.File(sample, "r") as f:
    vol = f["volume"][:].astype(np.float32)
    valid = f["valid_ascan_mask"][:]
    lat = f.attrs.get("laterality", "OD")
    lat = lat.decode() if isinstance(lat, bytes) else lat
res = process_volume(transform_volume(vol, valid), lat, valid_mask=valid)
out = res["volume"]; vm = res["valid_mask_256"]
sel = out[np.broadcast_to(vm[:, None, :], out.shape)]
check("單顆樣本 np.mean 落在 [0,1] 且 std>0",
      0 <= sel.mean() <= 1 and sel.std() > 0,
      f"該顆 mean={sel.mean():.4f} std={sel.std():.4f} shape={out.shape}")
check("輸出 shape == (25,256,256)", out.shape == (25, 256, 256), str(out.shape))
check("輸出值域在 [0,1]", out.min() >= 0 and out.max() <= 1.0001,
      f"[{out.min():.3f},{out.max():.3f}]")

# 有效像素比例 sanity: valid_mask_256 比例應略低於原生 valid_ascan_ratio (all-valid 降採更嚴)
vr_train = man.loc[(man["split"] == "train") & man["qc_pass"], "valid_ascan_ratio"].mean()
check("M5 有效像素比例 <= 平均 valid_ascan_ratio (all-valid 降採更嚴, 合理)",
      stats["valid_pixel_fraction"] <= vr_train + 1e-6,
      f"M5={stats['valid_pixel_fraction']*100:.2f}% vs 平均native={vr_train*100:.2f}%")

print("\n" + "=" * 70)
print(("全部通過 " + PASS) if ok_all else ("有檢查未通過 " + FAIL))
print("=" * 70)
sys.exit(0 if ok_all else 1)
