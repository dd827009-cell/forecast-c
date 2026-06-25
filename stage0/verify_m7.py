"""獨立驗證 M7 packer 是否正確。

流程: 用 m7_pack.py 把 manifest 打包到暫存目錄 → 獨立(不靠 packer 內部斷言)交叉檢查 →
預設清理暫存。檢查涵蓋: 完整性(每顆恰好打包一次/無重複 key)、shard 結構(vol/vmask/json)、
per-sample 與 per-shard sha256、值域/形狀、round-trip(從 h5 重算 M2+M3 的 sha 與 shard 內一致)、
版本感知增量(重跑 0 新增)。

用法: python stage0/verify_m7.py            # 預設 pilot manifest, 暫存後清理
      python stage0/verify_m7.py --keep     # 保留暫存 shard 供人工查看
"""
import argparse
import glob
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import h5py

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m2_transform import transform_volume          # noqa: E402
from m3_geometry import process_volume              # noqa: E402
from _version import STAGE0_VERSION                 # noqa: E402

PASS, FAIL = "✓", "✗"
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  -> {detail}" if detail else ""))


def read_member(tar, name):
    return tar.extractfile(name).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.parquet"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "stage1", "_verify_m7"))
    ap.add_argument("--vols-per-shard", type=int, default=16)
    ap.add_argument("--n-roundtrip", type=int, default=4)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    out = os.path.abspath(args.out_dir)
    if os.path.exists(out):
        shutil.rmtree(out)

    print("=" * 70)
    print("M7 packer 驗證 (打包暫存 → 獨立交叉比對)")
    print("=" * 70)

    # ---- 跑 packer (subprocess, 測真實 CLI) ----
    cmd = [sys.executable, os.path.join(HERE, "m7_pack.py"),
           "--manifest", args.manifest, "--out-dir", out,
           "--vols-per-shard", str(args.vols_per_shard), "--workers", "4"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    check("packer 執行成功 (exit 0)", r.returncode == 0, r.stderr.strip()[-200:] if r.returncode else "")
    if r.returncode != 0:
        print(r.stdout[-500:]); sys.exit(1)

    man = pd.read_parquet(args.manifest)
    man = man[man["qc_pass"]].copy()
    man["key"] = man.apply(lambda x: f"{x['patient_id']}__{x['eye']}__{x['visit_id']}", axis=1)
    expected = man[man["split"].isin(["train", "val", "test"])]
    exp_keys = set(expected["key"])

    # ---- manifest_packed 完整性 ----
    mp = pd.read_parquet(os.path.join(out, "manifest_packed.parquet"))
    packed_keys = list(mp["key"])
    check("manifest_packed 無重複 key", len(packed_keys) == len(set(packed_keys)),
          f"{len(packed_keys)} 列 / {len(set(packed_keys))} 唯一")
    check("打包 key 集合 == manifest qc_pass(train/val/test)", set(packed_keys) == exp_keys,
          f"packed {len(set(packed_keys))} vs expected {len(exp_keys)}")
    check("所有 packed row 標當前 transform_version",
          (mp["transform_version"] == STAGE0_VERSION).all())

    # ---- 逐 shard: 結構 + 檔級 sha + 收集 key ----
    shards = sorted(glob.glob(os.path.join(out, "octcube-*.tar")))
    check("有產生 shard", len(shards) > 0, f"{len(shards)} 個")
    all_keys, bad_struct, bad_shard_sha = [], 0, 0
    for sp in shards:
        side = json.load(open(sp + ".json", encoding="utf-8"))
        h = hashlib.sha256(open(sp, "rb").read()).hexdigest()
        if h != side["sha256"]:
            bad_shard_sha += 1
        with tarfile.open(sp) as t:
            names = t.getnames()
        by_key = {}
        for n in names:
            k, ext = n.split(".", 1)
            by_key.setdefault(k, set()).add(ext)
        for k, exts in by_key.items():
            all_keys.append(k)
            has_vol = any(e.startswith("vol.npy") for e in exts)
            if not (has_vol and "vmask.npy" in exts and "json" in exts):
                bad_struct += 1
    check("每個 shard 檔級 sha256 與 sidecar 相符", bad_shard_sha == 0, f"{bad_shard_sha} 個不符")
    check("每筆 sample 結構完整 (vol+vmask+json)", bad_struct == 0, f"{bad_struct} 筆缺件")
    check("shard 內 key 總數 == manifest_packed", len(all_keys) == len(packed_keys),
          f"{len(all_keys)} vs {len(packed_keys)}")
    check("shard 內 key 無重複 (無跨 shard 重複)", len(all_keys) == len(set(all_keys)))

    # ---- 抽樣: per-sample sha + 值域/形狀 + round-trip(從 h5 重算) ----
    sample_keys = packed_keys[:args.n_roundtrip]
    key2shard = {}
    for sp in shards:
        side = json.load(open(sp + ".json", encoding="utf-8"))
        for k in side["keys"]:
            key2shard[k] = (sp, side["ext"], side["compress"])
    n_sha_ok = n_rt_ok = n_vmask_ok = 0
    rng_ok = True
    for k in sample_keys:
        sp, ext, comp = key2shard[k]
        with tarfile.open(sp) as t:
            raw = read_member(t, k + ext)
            meta = json.loads(read_member(t, k + ".json"))
            vmask = np.load(io.BytesIO(read_member(t, k + ".vmask.npy")))
        npy = gzip.decompress(raw) if ext.endswith(".gz") else raw
        arr = np.load(io.BytesIO(npy))
        # per-sample sha (對未壓縮 npy bytes)
        if hashlib.sha256(npy).hexdigest() == meta["sha256"]:
            n_sha_ok += 1
        if arr.shape == (25, 256, 256) and str(arr.dtype) == "float16" \
                and arr.min() >= 0 and arr.max() <= 1.0001:
            pass
        else:
            rng_ok = False
        if vmask.shape == (25, 256) and vmask.dtype == bool:
            n_vmask_ok += 1
        # round-trip: 從 h5 重算 M2+M3 → fp16, sha 應與 shard 內一致
        h5p = man.loc[man["key"] == k, "h5_path"].iloc[0]
        with h5py.File(h5p, "r") as f:
            vol = f["volume"][:].astype(np.float32)
            valid = f["valid_ascan_mask"][:] if "valid_ascan_mask" in f else None
            lat = f.attrs.get("laterality", "OD")
            lat = lat.decode() if isinstance(lat, bytes) else lat
        v16 = np.ascontiguousarray(
            process_volume(transform_volume(vol, valid), lat, valid_mask=valid)["volume"].astype(np.float16))
        b = io.BytesIO(); np.save(b, v16)
        if hashlib.sha256(b.getvalue()).hexdigest() == meta["sha256"]:
            n_rt_ok += 1
    check(f"抽樣 {len(sample_keys)} 筆 per-sample sha256 一致", n_sha_ok == len(sample_keys),
          f"{n_sha_ok}/{len(sample_keys)}")
    check("抽樣 vol shape=(25,256,256) fp16 值域[0,1]", rng_ok)
    check(f"抽樣 vmask (25,256) bool", n_vmask_ok == len(sample_keys), f"{n_vmask_ok}/{len(sample_keys)}")
    check(f"round-trip: 從 h5 重算 M2+M3 的 sha 與 shard 一致 (packer 產出正確)",
          n_rt_ok == len(sample_keys), f"{n_rt_ok}/{len(sample_keys)}")

    # ---- 版本感知增量: 重跑應 0 新增 ----
    r2 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    check("增量重跑無新增 (版本相同全跳過)",
          "沒有待打包" in r2.stdout or "待打包 0 顆" in r2.stdout,
          r2.stdout.strip().splitlines()[-1][:80] if r2.stdout else "")
    n_shards_after = len(glob.glob(os.path.join(out, "octcube-*.tar")))
    check("增量重跑未新增 shard 檔", n_shards_after == len(shards), f"{n_shards_after} vs {len(shards)}")

    if not args.keep:
        shutil.rmtree(out)
        print(f"\n(已清理暫存 {out}; --keep 可保留)")
    else:
        print(f"\n暫存保留於 {out}")

    print("\n" + "=" * 70)
    print(("全部通過 " + PASS) if ok_all else ("有檢查未通過 " + FAIL))
    print("=" * 70)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
