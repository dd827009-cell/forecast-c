"""
Stage 0 - Module 7: 打包成 WebDataset tar shard (供 L40 從 NAS 串流訓練)。

對 manifest 的每顆 qc_pass volume 跑「與訓練相同的前處理」(M2 強度轉換 + M3 幾何, 變體A:
FOV置中裁512→resize256 + OS翻轉), 存成 float16 (25,256,256) [0,1], 打包進 tar shard。
**不做正規化、不內插 25→60**——那兩步留給 dataloader (改參數不必重打包)。

設計:
  - 格式 WebDataset tar (大塊順序讀, NAS 串流友善); 用 stdlib tarfile, 不依賴 webdataset 套件。
  - 每筆 sample = {key}.vol.npy(.gz) + {key}.vmask.npy + {key}.json; key = patient__eye__visit。
  - 依 split 分 shard: octcube-{split}-{NNNNNN}.tar, 每 shard --vols-per-shard 顆。
  - **精簡**: volume(fp16 256/25) + valid_mask_256(bool, ~6KB) + meta。
    層/IR/pos 屬 Stage2/COEP, 延後做 M7b (真相仍在原始 .h5)。
  - **增量**: 讀既有 manifest_packed.parquet 跳過已打包的 key; 新 shard 接續編號, 不重打包舊的。
  - **原子寫**: 先寫 .tmp 再 rename; 每 shard 附 .json sidecar 列出 keys。
  - **可攜**: --raw-root 可重寫 h5 路徑前綴 (manifest 的 h5_path 換機器時用); 純 numpy/h5py/cv2 無 torch。
  - **壓縮**: --compress {none,zlib}; OCT 背景 exact 0, zlib 壓縮率高 (NAS 頻寬↔CPU 取捨)。

用法:
    python stage0/m7_pack.py --manifest stage0/manifest.parquet --out-dir shards/ --workers 16
    python stage0/m7_pack.py ... --compress zlib --vols-per-shard 512
    # 增量: 重跑同指令即可 (自動跳過 manifest_packed 已記錄的)
"""
import argparse
import glob
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import h5py

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_transform import transform_volume          # noqa: E402
from m3_geometry import process_volume              # noqa: E402
from _version import STAGE0_VERSION                 # noqa: E402

META_FIELDS = ["patient_id", "eye", "visit_id", "longitudinal_key", "split",
               "H_orig", "W_orig", "image_quality", "valid_ascan_ratio", "age", "sex",
               "scale_axial_um_per_px", "scale_lateral_mm_per_px", "scale_bscan_spacing_mm",
               "eye_n_visits"]


def make_key(r):
    return f"{r['patient_id']}__{r['eye']}__{r['visit_id']}"


def _resolve_h5(path, raw_root):
    """--raw-root 給定時, 把 manifest 的 h5_path basename 接到新根 (換機器/掛載點)。"""
    if not raw_root:
        return path
    # 嘗試保留 <patient>/<file>.h5 兩層; 否則只接 basename
    parts = re.split(r"[\\/]", str(path))
    tail = os.path.join(*parts[-2:]) if len(parts) >= 2 else parts[-1]
    cand = os.path.join(raw_root, tail)
    return cand if os.path.exists(cand) else os.path.join(raw_root, parts[-1])


def process_one(task):
    """讀 h5 → M2+M3(變體A) → fp16(25,256,256)[0,1] → (key, vol_bytes, meta_bytes, err)。"""
    key, h5_path, raw_root, meta, compress = task
    try:
        p = _resolve_h5(h5_path, raw_root)
        with h5py.File(p, "r") as f:
            vol = f["volume"][:].astype(np.float32)
            valid = f["valid_ascan_mask"][:] if "valid_ascan_mask" in f else None
            lat = f.attrs.get("laterality", meta.get("eye", "OD"))
            lat = lat.decode() if isinstance(lat, bytes) else lat
        v01 = transform_volume(vol, valid)                       # M2
        res = process_volume(v01, lat, valid_mask=valid)         # M3 變體A (含 OS 翻轉)
        v256 = res["volume"]
        v16 = np.ascontiguousarray(v256.astype(np.float16))
        assert v16.shape == (25, 256, 256)
        buf = io.BytesIO(); np.save(buf, v16); npy = buf.getvalue()
        sha = hashlib.sha256(npy).hexdigest()            # #3: 對「未壓縮」陣列內容算雜湊
        raw = gzip.compress(npy, compresslevel=6) if compress == "zlib" else npy
        # #5: valid_mask_256 (bool (25,256), ~6KB)。缺則全 True。供未來 masking-aware loss。
        vm = res.get("valid_mask_256")
        if vm is None:
            vm = np.ones((v16.shape[0], v16.shape[2]), dtype=bool)
        vmbuf = io.BytesIO(); np.save(vmbuf, np.ascontiguousarray(vm.astype(bool)))
        vmask_bytes = vmbuf.getvalue()
        m = dict(meta)
        m.update({"key": key, "shape": list(v16.shape), "dtype": "float16",
                  "value_range": "[0,1]", "n_frames_stored": 25,
                  "transform_version": STAGE0_VERSION, "sha256": sha,
                  "has_valid_mask": True, "valid_mask_shape": list(vm.shape)})
        meta_bytes = json.dumps(m, ensure_ascii=False).encode("utf-8")
        return key, raw, meta_bytes, vmask_bytes, None
    except Exception as e:
        return key, None, None, None, f"{type(e).__name__}: {e}"


def _add(tar, name, data):
    ti = tarfile.TarInfo(name); ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))


def scan_existing(out_dir):
    """回傳 (next_idx_per_split, packed_keys, stale_keys)。
    #1 版本感知: packed = manifest_packed 中「transform_version 與當前相同」的 key;
    stale = 版本不同的 key (本次會重打包, 但會警告舊 shard 殘留)。"""
    next_idx = {}
    for f in glob.glob(os.path.join(out_dir, "octcube-*.tar")):
        m = re.search(r"octcube-(\w+)-(\d+)\.tar$", os.path.basename(f))
        if m:
            sp, idx = m.group(1), int(m.group(2))
            next_idx[sp] = max(next_idx.get(sp, -1), idx)
    next_idx = {sp: i + 1 for sp, i in next_idx.items()}
    packed, stale = set(), set()
    mp = os.path.join(out_dir, "manifest_packed.parquet")
    if os.path.exists(mp):
        dfp = pd.read_parquet(mp)
        if "transform_version" in dfp.columns:
            packed = set(dfp.loc[dfp["transform_version"] == STAGE0_VERSION, "key"])
            stale = set(dfp.loc[dfp["transform_version"] != STAGE0_VERSION, "key"])
        else:
            packed = set(dfp["key"])
    return next_idx, packed, stale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="stage0/manifest.parquet")
    ap.add_argument("--out-dir", default="shards")
    ap.add_argument("--raw-root", default=None, help="換機器時重寫 h5_path 的根 (預設用 manifest 內路徑)")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--vols-per-shard", type=int, default=512)
    ap.add_argument("--compress", choices=["none", "zlib"], default="none")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="#2: 給定則打包前先隨機打散樣本→shard (固定 seed 可重現); 不給=按 manifest 順序")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_parquet(args.manifest)
    df = df[df["qc_pass"]].copy()
    df["key"] = df.apply(make_key, axis=1)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    next_idx, packed, stale = scan_existing(args.out_dir)
    if stale:
        print(f"[警告][#1] 偵測到 {len(stale)} 個 key 是用『不同 transform_version』打包的。"
              f"本次會以當前版本 {STAGE0_VERSION} 重打包它們到新 shard，"
              f"但舊 shard 仍含舊版資料 → 同 key 將重複。"
              f"做版本升級時請改用全新 --out-dir，別在舊目錄上混。")
    todo = df[df["split"].isin(splits) & ~df["key"].isin(packed)]
    if args.shuffle_seed is not None:
        todo = todo.sample(frac=1.0, random_state=args.shuffle_seed).reset_index(drop=True)
        print(f"[#2] 打包前已洗牌 (seed={args.shuffle_seed})")
    print(f"manifest qc_pass {len(df)} 顆; 目標 split {splits}; 已打包(同版本) {len(packed)}; "
          f"待打包 {len(todo)} 顆; 壓縮={args.compress}; workers={args.workers}")
    if todo.empty:
        print("沒有待打包的 volume (全部已是最新)。"); return

    ext = ".vol.npy.gz" if args.compress == "zlib" else ".vol.npy"
    packed_rows, errs = [], []
    raw_bytes_tot = 0

    for sp in splits:
        sub = todo[todo["split"] == sp]
        if sub.empty:
            continue
        idx = next_idx.get(sp, 0)
        tasks = [(r["key"], r["h5_path"], args.raw_root,
                  {k: (None if pd.isna(r.get(k)) else
                       (int(r[k]) if k in ("H_orig", "W_orig", "eye_n_visits") and pd.notna(r.get(k))
                        else r[k])) for k in META_FIELDS},
                  args.compress) for _, r in sub.iterrows()]
        print(f"\n[{sp}] 打包 {len(tasks)} 顆 → shard 從 {idx:06d} 起 ({args.vols_per_shard}/shard)")

        def flush(samples, shard_idx):
            """原子寫一個 shard + sidecar; 回傳 shard 檔名。"""
            name = f"octcube-{sp}-{shard_idx:06d}.tar"
            path = os.path.join(args.out_dir, name); tmp = path + ".tmp"
            with tarfile.open(tmp, "w") as tar:
                for key, raw, meta_bytes, vmask_bytes in samples:
                    _add(tar, key + ext, raw)
                    _add(tar, key + ".vmask.npy", vmask_bytes)   # #5
                    _add(tar, key + ".json", meta_bytes)
            os.replace(tmp, path)
            # #3: shard 檔級 sha256 (讀取端可驗整顆 shard 沒在 NAS 傳輸中損壞)
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            keys = [s[0] for s in samples]
            with open(os.path.join(args.out_dir, name + ".json"), "w", encoding="utf-8") as fp:
                json.dump({"shard": name, "split": sp, "n": len(keys), "keys": keys,
                           "compress": args.compress, "ext": ext,
                           "sha256": h.hexdigest(), "transform_version": STAGE0_VERSION},
                          fp, ensure_ascii=False)
            return name

        buffer, pool = [], (Pool(args.workers) if args.workers > 1 else None)
        results = (pool.imap(process_one, tasks, chunksize=4) if pool
                   else map(process_one, tasks))
        n_done = 0
        for key, raw, meta_bytes, vmask_bytes, err in results:
            if err:
                errs.append((key, err)); continue
            raw_bytes_tot += len(raw)
            buffer.append((key, raw, meta_bytes, vmask_bytes))
            # 從 key 反查該列 meta 以寫 manifest_packed
            row = sub[sub["key"] == key].iloc[0]
            packed_rows.append({**{k: row.get(k) for k in (["key"] + META_FIELDS)},
                                "shard": f"octcube-{sp}-{idx:06d}.tar",
                                "h5_path": row["h5_path"],
                                "transform_version": STAGE0_VERSION})
            if len(buffer) >= args.vols_per_shard:
                nm = flush(buffer, idx); n_done += len(buffer)
                print(f"    寫出 {nm} ({len(buffer)} 顆, 累計 {n_done})")
                buffer = []; idx += 1
        if buffer:
            nm = flush(buffer, idx); n_done += len(buffer)
            print(f"    寫出 {nm} ({len(buffer)} 顆, 累計 {n_done})")
            idx += 1
        if pool:
            pool.close(); pool.join()
        next_idx[sp] = idx

    # 更新 manifest_packed (append)
    mp_path = os.path.join(args.out_dir, "manifest_packed.parquet")
    new_mp = pd.DataFrame(packed_rows)
    if os.path.exists(mp_path):
        new_mp = pd.concat([pd.read_parquet(mp_path), new_mp], ignore_index=True)
    new_mp.to_parquet(mp_path, index=False)

    # 摘要
    print(f"\n=== M7 摘要 ===")
    print(f"本次打包 {len(packed_rows)} 顆, 失敗 {len(errs)}")
    if raw_bytes_tot and len(packed_rows):
        print(f"平均每顆 shard 內位元組: {raw_bytes_tot/len(packed_rows)/1024:.0f} KB "
              f"({'壓縮後' if args.compress!='none' else '未壓縮'})")
    print(f"manifest_packed: {mp_path} (共 {len(new_mp)} 列)")
    for f in sorted(glob.glob(os.path.join(args.out_dir, "octcube-*.tar"))):
        sz = os.path.getsize(f) / 1024 / 1024
        print(f"  {os.path.basename(f)}: {sz:.1f} MB")
    if errs:
        print("錯誤範例:", errs[:3])


if __name__ == "__main__":
    main()
