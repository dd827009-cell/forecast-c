"""
Stage 0 - Module 7b: 抽存「評估/COEP 資產」成隨機存取格式 (供 Stage 2 健檢 + 未來 COEP)。

與 M7 的分工:
  - M7  = 給「訓練」吃的: 順序串流的 WebDataset tar, 只含 volume(fp16 256/25)+valid_mask_256。
  - M7b = 給「評估」查的: 隨機存取的 per-eye 檔, 含 M7 故意略過的 層/IR/pos/512-mask。
  兩者共用同一把 key (patient__eye__visit), Stage 2 用 manifest 的 eye_n_visits 挑縱向眼後,
  照 key 直接撈對應厚度/座標。真相仍在原始 .h5, M7b 只是加速層, 可隨時重跑。

存什麼 (全部來自 M3 process_volume 已算好的輸出, M7 沒撿; ir 另從原始 .h5 raw 讀):
  - ilm / rpe           (25,512) float32  方案C native (496列空間/512欄, 不 resize); NaN 保留
  - ilm_valid/rpe_valid (25,512) bool      = isfinite, Stage2 只在有效層處算厚度
  - valid_mask_512      (25,512) bool      provenance, 對齊層座標
  - ascan_pos_ir        (25,512,2) float32 方案A flip+crop, IR 768x768 座標數值不動
  - ir                  (768,768) uint8    raw en-face, **不翻** (pos_frame='raw_ir_unflipped')

格式:
  - 預設 per-eye `.npz` (np.savez_compressed): out_dir/<split>/<key>.npz
    → 零索引、零額外依賴, Stage2 照 key 開檔即可隨機存取。
    20 萬顆小檔若對 inode 有壓力, 之後可轉 LMDB/Zarr (內容不變, 重跑成本低)。
  - 每顆附 sha256; 增量: 讀 manifest_assets.parquet 跳過已存的同版本 key。
  - 純 numpy/h5py/cv2, 無 torch。

⚠️ 全量前: 建議 QC(M4) 已帶 vol_h!=496 旗標 (文件【B5】), 避免非 496 高度掃描讓厚度換算靜默錯位。
   本腳本只跑 manifest 中 qc_pass 的眼。

用法:
    # pilot 先驗內容正確
    python stage0/m7b_pack.py --manifest stage0/manifest.parquet --out-dir assets/ --workers 8
    # 增量: 重跑同指令, 自動跳過已存
    python stage0/m7b_pack.py ... --splits val,test   # Stage2 其實只需 val/test
"""
import argparse
import glob
import hashlib
import io
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import h5py

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_geometry import process_volume                # noqa: E402
from _version import STAGE0_VERSION                   # noqa: E402

# 與 M7 一致的 meta 欄位 (含 µm 換算所需 scale_axial + 縱向配對所需 eye_n_visits)
META_FIELDS = ["patient_id", "eye", "visit_id", "longitudinal_key", "split",
               "H_orig", "W_orig", "image_quality", "valid_ascan_ratio", "age", "sex",
               "scale_axial_um_per_px", "scale_lateral_mm_per_px", "scale_bscan_spacing_mm",
               "eye_n_visits"]

# 寫進 npz 的資產 key (None 的會被略過不存)
ASSET_KEYS = ["ilm", "rpe", "ilm_valid", "rpe_valid",
              "valid_mask_512", "ascan_pos_ir", "ir"]


def make_key(r):
    return f"{r['patient_id']}__{r['eye']}__{r['visit_id']}"


def _resolve_h5(path, raw_root):
    """--raw-root 給定時, 把 manifest 的 h5_path 接到新根 (換機器/掛載點)。同 M7。"""
    if not raw_root:
        return path
    parts = re.split(r"[\\/]", str(path))
    tail = os.path.join(*parts[-2:]) if len(parts) >= 2 else parts[-1]
    cand = os.path.join(raw_root, tail)
    return cand if os.path.exists(cand) else os.path.join(raw_root, parts[-1])


def process_one(task):
    """讀 h5 → M3 幾何(只取層/pos/mask, 丟棄 resize 後的 volume) + raw ir → 寫 npz。
    回傳 (key, out_path, sha256, err)。"""
    key, h5_path, raw_root, meta, out_dir, split, compress = task
    try:
        p = _resolve_h5(h5_path, raw_root)
        with h5py.File(p, "r") as f:
            vol = f["volume"][:].astype(np.float32)          # 只為了 shape/OS-flip/crop, 不存
            lat = f.attrs.get("laterality", meta.get("eye", "OD"))
            lat = lat.decode() if isinstance(lat, bytes) else lat
            ilm = f["ilm_y"][:].astype(np.float32) if "ilm_y" in f else None
            rpe = f["rpe_bm_y"][:].astype(np.float32) if "rpe_bm_y" in f else None
            valid = f["valid_ascan_mask"][:] if "valid_ascan_mask" in f else None
            pos = f["ascan_pos_ir"][:].astype(np.float32) if "ascan_pos_ir" in f else None
            ir = f["ir"][:] if "ir" in f else None           # raw, 不翻

        # M3 幾何: 與 M7 完全相同的 flip+crop 決策 (只看 W, 與強度無關), 故跳過 M2。
        res = process_volume(vol, lat, ilm=ilm, rpe=rpe, valid_mask=valid, pos=pos)

        out = {}
        for k in ["ilm", "rpe", "ilm_valid", "rpe_valid", "valid_mask_512", "ascan_pos_ir"]:
            if res.get(k) is not None:
                out[k] = res[k]
        if ir is not None:
            out["ir"] = np.ascontiguousarray(ir)             # raw_ir_unflipped

        # meta 以 0-d object array 夾帶 (Stage2 讀 npz 即拿得到 key/scale/eye_n_visits...)
        m = dict(meta)
        m.update({"key": key, "transform_version": STAGE0_VERSION,
                  "pos_frame": "raw_ir_unflipped",
                  "layer_space": "native_496row_512col_no_resize",
                  "stored": sorted(out.keys())})
        out["meta"] = np.array(json.dumps(m, ensure_ascii=False))

        # 序列化 → sha256 → 原子寫
        buf = io.BytesIO()
        (np.savez_compressed if compress else np.savez)(buf, **out)
        data = buf.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        path = os.path.join(split_dir, key + ".npz")
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        return key, path, sha, None
    except Exception as e:
        return key, None, None, f"{type(e).__name__}: {e}"


def scan_existing(out_dir):
    """版本感知增量: 回傳已存且同版本的 key 集合。"""
    packed = set()
    mp = os.path.join(out_dir, "manifest_assets.parquet")
    if os.path.exists(mp):
        dfp = pd.read_parquet(mp)
        if "transform_version" in dfp.columns:
            packed = set(dfp.loc[dfp["transform_version"] == STAGE0_VERSION, "key"])
        else:
            packed = set(dfp["key"])
    return packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="stage0/manifest.parquet")
    ap.add_argument("--out-dir", default="assets")
    ap.add_argument("--raw-root", default=None, help="換機器時重寫 h5_path 的根")
    ap.add_argument("--splits", default="train,val,test",
                    help="Stage2 健檢其實只需 val,test; 預設全切")
    ap.add_argument("--no-compress", action="store_true",
                    help="預設 savez_compressed; 給此旗標改用未壓縮 savez (省 CPU)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_parquet(args.manifest)
    df = df[df["qc_pass"]].copy()
    df["key"] = df.apply(make_key, axis=1)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    packed = scan_existing(args.out_dir)
    todo = df[df["split"].isin(splits) & ~df["key"].isin(packed)]
    print(f"manifest qc_pass {len(df)} 顆; 目標 split {splits}; 已存(同版本) {len(packed)}; "
          f"待抽存 {len(todo)} 顆; 壓縮={not args.no_compress}; workers={args.workers}")
    if todo.empty:
        print("沒有待抽存的眼 (全部已是最新)。")
        return

    compress = not args.no_compress
    tasks = []
    for _, r in todo.iterrows():
        meta = {k: (None if pd.isna(r.get(k)) else
                    (int(r[k]) if k in ("H_orig", "W_orig", "eye_n_visits") and pd.notna(r.get(k))
                     else r[k])) for k in META_FIELDS}
        tasks.append((r["key"], r["h5_path"], args.raw_root, meta,
                      args.out_dir, r["split"], compress))

    rows, errs = [], []
    pool = Pool(args.workers) if args.workers > 1 else None
    results = (pool.imap_unordered(process_one, tasks, chunksize=4) if pool
               else map(process_one, tasks))
    n = 0
    for key, path, sha, err in results:
        if err:
            errs.append((key, err))
            continue
        row = todo[todo["key"] == key].iloc[0]
        rows.append({**{k: row.get(k) for k in (["key"] + META_FIELDS)},
                     "asset_path": os.path.relpath(path, args.out_dir),
                     "sha256": sha, "transform_version": STAGE0_VERSION})
        n += 1
        if n % 200 == 0:
            print(f"    已抽存 {n} 顆...")
    if pool:
        pool.close()
        pool.join()

    # 更新 manifest_assets (append)
    mp_path = os.path.join(args.out_dir, "manifest_assets.parquet")
    new_mp = pd.DataFrame(rows)
    if os.path.exists(mp_path) and not new_mp.empty:
        new_mp = pd.concat([pd.read_parquet(mp_path), new_mp], ignore_index=True)
    if not new_mp.empty:
        new_mp.to_parquet(mp_path, index=False)

    print(f"\n=== M7b 摘要 ===")
    print(f"本次抽存 {len(rows)} 顆, 失敗 {len(errs)}")
    print(f"manifest_assets: {mp_path} (共 {len(new_mp)} 列)")
    if errs:
        print("錯誤範例:", errs[:3])


if __name__ == "__main__":
    main()
