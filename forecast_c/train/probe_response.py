"""Step 0 反應 probe：OCTCube 語意對「治療反應預測」有沒有加值 → gate A/B/C。

核心問題（配 docs 雙 latent + 使用者下游任務表）：
  從**當前 volume 特徵**預測**未來 ΔCST**，比較三組輸入——
    ① 只 z_thickness(B0 latent)   ② 只 OCTCube 某層   ③ 兩者合併
  三組都吃相同共變數(Δt、baseline CST，可選藥種)，所以比的是「特徵本身」。

判讀：
  - ② > ① → 語意確實幫治療預測(語意分支值得做) 且 z_thickness 有缺口(瓶頸在)。
  - ② ≈ ① → 語意沒用 or B0 已抓到 → 分支買不到東西 → A/B/C 全免，只用 B0。
  - ③ > 各自 → 互補 → fusion(z_visit) 站得住。
  - 順便：逐層(oct_b12..b24)看哪層對「反應」最有用 = 用真正終點回答 Q1(該蒸哪層)，
    取代之前測錯構念的「AMD 亞型分類」。
參考基準：
  - persistence(預測 ΔCST=0)：治療反應的招牌 baseline(存亡閘門 P0)。
  - cov-only(只 Δt+baseline)：影像特徵有沒有贏過「只看時間+起點厚度」。

不受「全是 AMD」限制、不用等醫師標註(ΔCST 從 h5 厚度 GT 直接算)。
這是拿縱向 label 當**評估**，不訓練 encoder → 不違背「單 volume 表徵先做完」。

═══════════════════════════════════════════════════════════════════════════════
 依賴本機介面(同 probe_semantics)：AnisotropicUNet3D.encode_multiscale + forward_layers。
 B0 前處理 prepare_b0_input 直接 import 自 probe_semantics（唯一要人工確認處在那）。
═══════════════════════════════════════════════════════════════════════════════

兩階段(可續跑):
  # 1) 抽 t-visit 特徵 + 所有 visit 的 CST（重，GPU+權重+資料）
  python -m forecast_c.train.probe_response --stage extract \
      --h5-dir /path/h5_output --b0-ckpt probe_out/anisotropic_native_sqrtpct_seed43/best.pt \
      --cache probe_out/response_probe_cache
  # 2) 擬合 probe（輕，只吃快取，可反覆調）
  python -m forecast_c.train.probe_response --stage probe \
      --cache probe_out/response_probe_cache --folds 5
  # 想加藥種共變數：--case-pooling /path/case_pooling.xlsx（extract 與 probe 都帶）
"""
import argparse
import collections
import datetime
import glob
import os

import numpy as np

from forecast_c.train.probe_semantics import (
    prepare_b0_input, _load_b0, _get_oct_taps, _pool_b0, _pool_oct,
    REQUESTED_BLOCKS, _group_kfold,
)

DRUGS = ["Eylea", "Lucentis", "Avastin"]          # case-pooling 多熱編碼順序


# ─────────────────────────────── 配對（純 h5）───────────────────────────────
def _parse_t(iso):
    try:
        return datetime.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except Exception:
        return None


def _visit_meta(path):
    import h5py
    with h5py.File(path, "r") as h:
        return {
            "h5": path,
            "key": str(h.attrs.get("longitudinal_key", "")),
            "pid": str(h.attrs.get("patient_id", "")).strip(),
            "lat": str(h.attrs.get("laterality", "")).strip(),
            "time": str(h.attrs.get("acquisition_time_utc", "")),
            "day": str(h.attrs.get("acquisition_time_utc", ""))[:10],
            "q": float(h.attrs.get("image_quality", 0.0) or 0.0),
        }


def build_pairs(h5_dir, all_pairs=False):
    """掃 h5 → 依 longitudinal_key 分眼 → 同日去重(留 image_quality 高) → 排序 → 配對。

    回 [(t_meta, future_meta), ...]。預設相鄰配對；all_pairs=所有 i<j。
    """
    groups = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(h5_dir, "**", "*.h5"), recursive=True)):
        try:
            m = _visit_meta(f)
        except Exception:
            continue
        if m["key"] and m["time"]:
            groups[m["key"]].append(m)
    pairs = []
    for visits in groups.values():
        byday = {}
        for v in visits:
            if v["day"] not in byday or v["q"] > byday[v["day"]]["q"]:
                byday[v["day"]] = v
        visits = sorted(byday.values(), key=lambda x: x["time"])
        if len(visits) < 2:
            continue
        if all_pairs:
            for i in range(len(visits)):
                for j in range(i + 1, len(visits)):
                    pairs.append((visits[i], visits[j]))
        else:
            for i in range(len(visits) - 1):
                pairs.append((visits[i], visits[i + 1]))
    return pairs


# ─────────────────────────────── 治療共變數（可選）───────────────────────────────
def load_drug_cov(case_pooling_xlsx):
    """case-pooling → {(病歷號,'OD'/'OS'): drug multihot np[3]}。沒給檔就回 None。"""
    if not case_pooling_xlsx:
        return None
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from pipeline.case_pooling import parse_case_pooling
    recs = parse_case_pooling(case_pooling_xlsx)
    out = {}
    for (chart, eye), r in recs.items():
        mh = np.array([1.0 if d in r.get("drugs", []) else 0.0 for d in DRUGS], dtype=np.float32)
        out[(str(chart).strip(), eye)] = mh
    return out


# ─────────────────────────────── 抽特徵 + CST（stage=extract）───────────────────────────────
def _cache_path(cache, h5_dir, h5_path):
    rel = os.path.relpath(h5_path, h5_dir).replace(os.sep, "__").replace("/", "__")
    return os.path.join(cache, rel[:-3] + ".npz")


def extract(args):
    import torch
    import h5py
    from forecast_c.model.encoder import build_octcube_encoder, OCTCubeTokenEncoder
    from forecast_c.data.oct_h5 import cst as cst_of

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    pairs = build_pairs(args.h5_dir, all_pairs=args.all_pairs)
    t_h5 = {t["h5"] for t, _ in pairs}                    # 只有 t-visit 要特徵
    all_h5 = t_h5 | {f["h5"] for _, f in pairs}           # 所有端點都要 CST
    os.makedirs(args.cache, exist_ok=True)
    print(f"device={dev}  配對={len(pairs)}  眼={len({t['key'] for t,_ in pairs})}  "
          f"需特徵眼次={len(t_h5)}  需CST眼次={len(all_h5)}")
    if not pairs:
        print("⚠️ 0 配對 → 檢查 h5 的 longitudinal_key / acquisition_time_utc")
        return

    print("載入 B0 ...")
    b0 = _load_b0(args.b0_ckpt, args.b0_base_channels, dev)
    print("載入凍結 OCTCube ...")
    oct_enc = build_octcube_encoder(ckpt_path=args.octcube_ckpt, repo_dir=args.repo_dir,
                                    device=dev, freeze=False)
    oct_inner = getattr(oct_enc, "inner", oct_enc)
    oct_inner.eval()

    done = skipped = failed = 0
    for i, path in enumerate(sorted(all_h5)):
        need_feat = path in t_h5
        outp = _cache_path(args.cache, args.h5_dir, path)
        if os.path.exists(outp):
            z = np.load(outp)
            if "cst" in z.files and (not need_feat or "b0_latent" in z.files):
                skipped += 1
                continue
        try:
            store = {}
            store["cst"] = np.float32(cst_of(path))
            if need_feat:
                with h5py.File(path, "r") as h:
                    vol = h["volume"][:]
                with torch.no_grad():
                    ms = b0.encode_multiscale(prepare_b0_input(vol).to(dev))
                    store["b0_latent"] = _pool_b0(ms["latent"]).astype(np.float32)
                    store["b0_level3"] = _pool_b0(ms["level3"]).astype(np.float32)
                    taps = _get_oct_taps(oct_inner, OCTCubeTokenEncoder.prep(vol).to(dev))
                    for blk in REQUESTED_BLOCKS:
                        store[f"oct_b{blk}"] = _pool_oct(taps[blk]).astype(np.float32)
            np.savez(outp, **store)
            done += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {path}: {type(e).__name__}: {e}")
        if (i + 1) % 20 == 0 or i + 1 == len(all_h5):
            print(f"  {i+1}/{len(all_h5)}  new={done} skip={skipped} fail={failed}")
    print(f"抽取完成：new={done} skip={skipped} fail={failed} → {args.cache}")


# ─────────────────────────────── ridge + 指標（純 numpy）───────────────────────────────
def _fit_predict(Xtr, ytr, Xte, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6               # standardize：train 統計，吸收各層/共變數尺度
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    d = Xtr.shape[1]
    w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(d), Xtr.T @ (ytr - ym))
    return Xte @ w + ym                                   # 截距靠 y 置中處理，不懲罰


def _r2(y, yh):
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yh) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def _mae(y, yh):
    return float(np.abs(y - yh).mean())


# ─────────────────────────────── probe（stage=probe）───────────────────────────────
def _build_rows(args):
    """回 (rows, drug_cov)。每 row = {feat...,'dcst','dt','base','pid','drug'}。缺特徵/NaN CST 的丟掉。"""
    pairs = build_pairs(args.h5_dir, all_pairs=args.all_pairs) if args.h5_dir else None
    if pairs is None:
        raise SystemExit("probe 需要 --h5-dir 來重建配對(輕量，只讀 attrs)")
    drug_cov = load_drug_cov(args.case_pooling)
    cache = {}

    def _load(path):
        if path not in cache:
            p = _cache_path(args.cache, args.h5_dir, path)
            cache[path] = np.load(p) if os.path.exists(p) else None
        return cache[path]

    rows = []
    miss = 0
    for t, f in pairs:
        zt, zf = _load(t["h5"]), _load(f["h5"])
        if zt is None or zf is None or "b0_latent" not in getattr(zt, "files", []):
            miss += 1
            continue
        ct, cf = float(zt["cst"]), float(zf["cst"])
        tt, tf = _parse_t(t["time"]), _parse_t(f["time"])
        if not (np.isfinite(ct) and np.isfinite(cf)) or tt is None or tf is None:
            miss += 1
            continue
        row = {"dcst": cf - ct, "base": ct, "dt": (tf - tt).days / 365.25, "pid": t["pid"]}
        row["b0_latent"] = zt["b0_latent"]
        row["b0_level3"] = zt["b0_level3"] if "b0_level3" in zt.files else None
        for blk in REQUESTED_BLOCKS:
            row[f"oct_b{blk}"] = zt[f"oct_b{blk}"]
        if drug_cov is not None:
            row["drug"] = drug_cov.get((t["pid"], t["lat"]), np.zeros(len(DRUGS), np.float32))
        rows.append(row)
    print(f"可用配對={len(rows)}（丟棄缺特徵/NaN CST {miss}）")
    return rows, drug_cov


def probe(args):
    rows, drug_cov = _build_rows(args)
    if not rows:
        print("⚠️ 無可用配對 → 先跑 --stage extract")
        return
    pids = [r["pid"] for r in rows]
    y = np.array([r["dcst"] for r in rows], dtype=np.float64)
    cov = np.array([[r["dt"], r["base"]] for r in rows], dtype=np.float64)
    if drug_cov is not None:
        cov = np.hstack([cov, np.stack([r["drug"] for r in rows])])
    print(f"\n配對={len(y)}  病人={len(set(pids))}  ΔCST µm: mean={y.mean():.1f} std={y.std():.1f} "
          f"|Δ|mean={np.abs(y).mean():.1f}  folds={args.folds}\n")

    # 要比較的輸入組（都串共變數）
    sets = {"cov-only": None, "b0(z_thickness)": "b0_latent", "b0_level3": "b0_level3"}
    for blk in REQUESTED_BLOCKS:
        sets[f"oct_b{blk}"] = f"oct_b{blk}"
    sets["b0+oct_b18(combined)"] = ("b0_latent", "oct_b18")

    def _mat(key):
        if key is None:
            return cov
        keys = key if isinstance(key, tuple) else (key,)
        parts = [np.stack([r[k] for r in rows]) for k in keys]
        return np.hstack(parts + [cov])

    folds = _group_kfold(pids, args.folds, seed=args.seed)
    # persistence 參考（預測 ΔCST=0）
    pers_mae = float(np.mean([_mae(y[te], np.zeros(len(te))) for _, te in folds]))
    print(f"{'input':<24}{'dim':>5}  {'R2':>14}{'MAE(µm)':>14}")
    print("-" * 57)
    print(f"{'persistence(Δ=0)':<24}{'-':>5}  {'-':>14}{pers_mae:>14.2f}")

    results = {}
    for name, key in sets.items():
        X = _mat(key)
        r2s, maes = [], []
        for tr, te in folds:
            yh = _fit_predict(X[tr], y[tr], X[te], args.alpha)
            r2s.append(_r2(y[te], yh))
            maes.append(_mae(y[te], yh))
        r2m, r2s_ = float(np.mean(r2s)), float(np.std(r2s))
        maem, maes_ = float(np.mean(maes)), float(np.std(maes))
        results[name] = (r2m, maem)
        print(f"{name:<24}{X.shape[1]:>5}  {r2m:>7.3f}±{r2s_:.3f}{maem:>9.2f}±{maes_:.2f}")

    # ── 判讀 ──
    print("\n── 判讀 ──")
    b0_r2 = results["b0(z_thickness)"][0]
    cov_r2 = results["cov-only"][0]
    oct_r2 = {b: results[f"oct_b{b}"][0] for b in REQUESTED_BLOCKS}
    best_blk = max(oct_r2, key=oct_r2.get)
    comb_r2 = results["b0+oct_b18(combined)"][0]
    print(f"影像 vs 純共變數：b0 R²={b0_r2:.3f} vs cov-only {cov_r2:.3f} "
          f"→ 影像{'有' if b0_r2 > cov_r2 + 0.01 else '幾乎沒'}加值")
    print(f"Q1(對反應)最佳 OCTCube 層：oct_b{best_blk} R²={oct_r2[best_blk]:.3f}"
          f"（峰值{'在' if best_blk == 24 else '不在'} Block24）")
    gap = oct_r2[best_blk] - b0_r2
    if gap > 0.02:
        v = "語意有加值且 z_thickness 有缺口 → 語意分支值得做、瓶頸存在"
    elif gap < -0.02:
        v = "z_thickness 反而更好 → OCTCube 對反應沒帶新東西 → 傾向只用 B0"
    else:
        v = "打平 → 語意對反應沒用 or B0 已抓到 → 分支買不到東西，A/B/C 可全免"
    print(f"Q_value：oct 最佳 {oct_r2[best_blk]:.3f} − b0 {b0_r2:.3f} = {gap:+.3f} → {v}")
    if comb_r2 > max(b0_r2, oct_r2[best_blk]) + 0.01:
        print(f"合併 R²={comb_r2:.3f} > 各自 → 互補，fusion(z_visit) 站得住")


def main():
    ap = argparse.ArgumentParser(description="Step 0 反應 probe（ΔCST 回歸，gate A/B/C）")
    ap.add_argument("--stage", choices=["extract", "probe", "both"], default="both")
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--case-pooling", default=None, help="給了就把藥種多熱當共變數（可選）")
    ap.add_argument("--b0-ckpt", default="probe_out/anisotropic_native_sqrtpct_seed43/best.pt")
    ap.add_argument("--b0-base-channels", type=int, default=16)
    ap.add_argument("--octcube-ckpt", default="ckpts/OCTCube.pth")
    ap.add_argument("--repo-dir", default="OCTCubeM-main/OCTCube")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--all-pairs", action="store_true", help="所有 i<j 配對（預設只相鄰）")
    ap.add_argument("--cache", default="probe_out/response_probe_cache")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=10.0, help="ridge 正則")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.stage in ("extract", "both"):
        extract(a)
    if a.stage in ("probe", "both"):
        probe(a)


if __name__ == "__main__":
    main()
