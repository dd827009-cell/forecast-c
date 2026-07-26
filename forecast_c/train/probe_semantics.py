"""語意保留 probe（Z1 / Z2 / Z3）：量「哪個 latent 保留較多 OCTCube 疾病語意」。

回答架構拍板前的兩個可量問題：
  Q1 目標對不對：OCTCube 的疾病訊號在第幾個 block 最高？(Block24 值不值得追)
  Q2 供得起嗎  ：B0 厚度 latent 帶多少疾病資訊？(z_semantic=f(z_thickness) 的天花板)

做法 = frozen linear / small-MLP probe，靶 = case-pooling 的 AMD 亞型三分類
       (1=nAMD / 2=PCV / 3=RAP)，病人層級 GroupKFold 防洩漏。

特徵來源（每眼一個池化向量）：
  Z1  oct_b12/16/18/20/22/24 : OCTCube forward_layers 逐層 token 平均池化 → [1024]
  Z2  b0_latent              : B0 features["latent"]  平均池化 → [128]   ← z_semantic 的資訊上限
  Z3  b0_level3              : B0 features["level3"]   平均池化 → [64]    ← 更早/更豐富的 B0 tap

判讀（配 docs/目前研究進度與雙Latent架構）：
  - b0_latent 分數 ≈ oct 最佳層 → 無瓶頸、synergy 強 → 「掛 B0」的雙 latent 成立、也偏好共訓(A)。
  - b0_latent 分數 << oct 最佳層 → 有瓶頸 → 看 b0_level3 有沒有救；沒救就得獨立 semantic 學生(C)。
  - oct 逐層看峰值在不在 Block24 → 決定 D3 到底該蒸哪幾層（別無腦追最深）。

═══════════════════════════════════════════════════════════════════════════════
 本機介面依賴（cloud repo 沒有，跑在你本機 / L40 的 octcube-dev）：
   - forecast_c.model.anisotropic_unet.AnisotropicUNet3D + .encode_multiscale()
   - forecast_c.model.encoder.OCTCubeTokenEncoder.forward_layers()（你本機已加）
 ⚠️ 唯一要人工確認的是 prepare_b0_input()：必須 byte-match B0 訓練前處理。見該函式 banner。
═══════════════════════════════════════════════════════════════════════════════

兩階段（都可續跑）:
  # 1) 抽特徵（重，需 GPU+權重+資料，跑一次快取）
  python -m forecast_c.train.probe_semantics --stage extract \
      --h5-dir /path/h5_output --case-pooling /path/case_pooling.xlsx \
      --b0-ckpt probe_out/anisotropic_native_sqrtpct_seed43/best.pt \
      --cache probe_out/semantics_probe_cache
  # 2) 擬合 probe（輕，只吃快取，可反覆調）
  python -m forecast_c.train.probe_semantics --stage probe \
      --cache probe_out/semantics_probe_cache --folds 5 --probe both
  # 若 PCV/RAP 太少：加 --binary 退成 nAMD vs rest（快取不用重抽）
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

# ── 靶：case-pooling 診斷編碼 → 類別 index ──
DX_MAP = {"1": 0, "2": 1, "3": 2}          # 1=nAMD, 2=PCV, 3=RAP
CLASS_NAMES = ["nAMD", "PCV", "RAP"]

# ── Z1：要抽的 OCTCube block（1-based）→ forward_layers 的 0-based index ──
REQUESTED_BLOCKS = [12, 16, 18, 20, 22, 24]
LAYER_INDICES = tuple(b - 1 for b in REQUESTED_BLOCKS)
OCT_FEATS = [f"oct_b{b}" for b in REQUESTED_BLOCKS]
B0_FEATS = ["b0_latent", "b0_level3"]
ALL_FEATS = OCT_FEATS + B0_FEATS


# ════════════════════════════════════════════════════════════════════════════
#  ⚠️⚠️⚠️  唯一要你/Codex 確認的地方  ⚠️⚠️⚠️
# ════════════════════════════════════════════════════════════════════════════
def prepare_b0_input(volume):
    """raw h5 volume (D,H,W) uint16 → B0 輸入 tensor [1,1,25,496,512]。

    ★ 必須與 B0 訓練前處理逐步一致，否則特徵分布改變、probe 數字無意義。
    目前實作 = 依你交接說明的 `sqrt_then_percentile, P1/P99.5`：
        to float → resize 到 (25,496,512) → sqrt → robust percentile(P1,P99.5) → clip[0,1]
    需確認兩點：
      (a) 若 W≠512（384/768 的眼）B0 訓練是怎麼到 512 的（trilinear? crop? pad?）—— 這裡用 trilinear。
      (b) sqrt 與 percentile 的先後、percentile 是 per-volume 還是 train 全域統計。
    若 anisotropic_unet.py 已把訓練 transform 抽成函式，**直接 import 它取代本函式**最保險。
    """
    import torch
    import torch.nn.functional as F
    x = torch.as_tensor(np.asarray(volume), dtype=torch.float32)
    if x.dim() == 3:
        x = x[None, None]                                   # (1,1,D,H,W)
    if tuple(x.shape[-3:]) != (25, 496, 512):
        x = F.interpolate(x, size=(25, 496, 512), mode="trilinear", align_corners=False)
    x = torch.sqrt(torch.clamp(x, min=0.0))
    lo = torch.quantile(x, 0.01)
    hi = torch.quantile(x, 0.995)
    x = torch.clamp((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return x
# ════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────── 標籤 ───────────────────────────────
def load_labels(case_pooling_xlsx):
    """case-pooling → {(病歷號, 'OD'/'OS'): class_idx}（只留 diagnosis ∈ {1,2,3}）。"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from pipeline.case_pooling import parse_case_pooling
    recs = parse_case_pooling(case_pooling_xlsx)
    labels = {}
    for (chart, eye), r in recs.items():
        dx = str(r.get("diagnosis", "")).strip()
        if dx in DX_MAP:
            labels[(str(chart).strip(), eye)] = DX_MAP[dx]
    return labels


# ─────────────────────────────── 掃 h5 ───────────────────────────────
def _h5_meta(path):
    import h5py
    with h5py.File(path, "r") as h:
        pid = str(h.attrs.get("patient_id", "")).strip()
        lat = str(h.attrs.get("laterality", "")).strip()
        q = float(h.attrs.get("image_quality", 0.0) or 0.0)
    return pid, lat, q


def select_one_per_eye(h5_dir, labels, all_visits=False):
    """掃 h5，join 標籤。回 [(path, pid, lat, label)]；預設每眼留 image_quality 最高的一張。

    baseline diagnosis 是 per-eye 常數 → 一眼一張最乾淨（避免多回診相關樣本灌水）。
    """
    best = {}                                               # (pid,lat) -> (q, path)
    picked = []
    for f in sorted(glob.glob(os.path.join(h5_dir, "**", "*.h5"), recursive=True)):
        try:
            pid, lat, q = _h5_meta(f)
        except Exception:
            continue
        key = (pid, lat)
        if key not in labels:
            continue
        if all_visits:
            picked.append((f, pid, lat, labels[key]))
        elif key not in best or q > best[key][0]:
            best[key] = (q, f)
    if not all_visits:
        picked = [(p, pid, lat, labels[(pid, lat)]) for (pid, lat), (_, p) in best.items()]
    return picked


# ─────────────────────────────── 池化 ───────────────────────────────
def _pool_b0(feat):
    """B0 特徵 [B,C,D,H,W] → 空間+深度平均 → np[C]（B=1）。"""
    return feat.float().mean(dim=(2, 3, 4))[0].cpu().numpy()


def _pool_oct(tokens):
    """OCTCube 某層 tokens [B,5120,1024] → token 平均 → np[1024]（B=1）。

    對齊 OCTCube 自己 global_pool=True 的讀出方式；跨層公平性由 probe 端 standardize 處理。
    """
    return tokens.float().mean(dim=1)[0].cpu().numpy()


# ─────────────────────────────── 抽特徵（stage=extract）───────────────────────────────
def _safe_name(pid, lat):
    return re.sub(r"[^0-9A-Za-z_.-]", "_", f"{pid}__{lat}")


def _load_b0(ckpt, base_channels, device):
    import torch
    from forecast_c.model.anisotropic_unet import AnisotropicUNet3D
    b0 = AnisotropicUNet3D(base_channels=base_channels)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("model", "state_dict", "b0", "net"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    b0.load_state_dict(sd)                                  # strict：對不上就早報錯，別默默錯位
    return b0.eval().to(device)


def _get_oct_taps(oct_inner, x):
    """呼叫 forward_layers 並統一成 {block(1-based): (tokens, cls)}。dict / list 兩種回傳都吃。"""
    taps = oct_inner.forward_layers(x, layer_indices=LAYER_INDICES)
    out = {}
    for pos, (blk, li) in enumerate(zip(REQUESTED_BLOCKS, LAYER_INDICES)):
        item = taps[li] if isinstance(taps, dict) else taps[pos]
        tokens = item[0] if isinstance(item, (tuple, list)) else item
        out[blk] = tokens
    return out


def extract(args):
    import torch
    import h5py
    from forecast_c.model.encoder import build_octcube_encoder, OCTCubeTokenEncoder

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    labels = load_labels(args.case_pooling)
    picked = select_one_per_eye(args.h5_dir, labels, all_visits=args.all_visits)
    if args.limit:
        picked = picked[:args.limit]
    os.makedirs(args.cache, exist_ok=True)

    cls_hist = np.bincount([lb for *_, lb in picked], minlength=len(CLASS_NAMES))
    print(f"device={dev}  可標註樣本={len(picked)}  類別分布(nAMD/PCV/RAP)={cls_hist.tolist()}")
    if len(picked) == 0:
        print("⚠️ 0 個 (病歷號,眼) join 到診斷 → 檢查 patient_id↔Chart no. / laterality↔OD/OS 是否同格式")
        return

    print("載入 B0 ...")
    b0 = _load_b0(args.b0_ckpt, args.b0_base_channels, dev)
    print("載入凍結 OCTCube ...")
    oct_enc = build_octcube_encoder(ckpt_path=args.octcube_ckpt, repo_dir=args.repo_dir,
                                    device=dev, freeze=False)
    oct_inner = getattr(oct_enc, "inner", oct_enc)          # freeze=False 應直接是 inner，保險再取一次
    oct_inner.eval()

    done = skipped = failed = 0
    for i, (path, pid, lat, label) in enumerate(picked):
        outp = os.path.join(args.cache, _safe_name(pid, lat) + ".npz")
        if os.path.exists(outp):
            skipped += 1
            continue
        try:
            with h5py.File(path, "r") as h:
                vol = h["volume"][:]
            feats = {}
            with torch.no_grad():
                b0_in = prepare_b0_input(vol).to(dev)
                ms = b0.encode_multiscale(b0_in)
                feats["b0_latent"] = _pool_b0(ms["latent"])
                feats["b0_level3"] = _pool_b0(ms["level3"])
                oct_in = OCTCubeTokenEncoder.prep(vol).to(dev)
                taps = _get_oct_taps(oct_inner, oct_in)
                for blk in REQUESTED_BLOCKS:
                    feats[f"oct_b{blk}"] = _pool_oct(taps[blk])
            np.savez(outp, label=np.int64(label), pid=pid, lat=lat,
                     **{k: v.astype(np.float32) for k, v in feats.items()})
            done += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {path}: {type(e).__name__}: {e}")
        if (i + 1) % 20 == 0 or i + 1 == len(picked):
            print(f"  {i+1}/{len(picked)}  new={done} skip={skipped} fail={failed}")
    print(f"抽取完成：new={done} skip={skipped} fail={failed} → {args.cache}")


# ─────────────────────────────── 指標（純 numpy）───────────────────────────────
def _rank_avg(x):
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _auc(y_bin, score):
    y_bin = y_bin.astype(bool)
    n_pos = int(y_bin.sum())
    n_neg = len(y_bin) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = _rank_avg(score)
    return (ranks[y_bin].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _metrics(y_true, y_pred, proba, n_classes):
    f1s, recalls, aucs, counts = [], [], [], []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        recalls.append(rec)
        aucs.append(_auc((y_true == c).astype(float), proba[:, c]))
        counts.append(int((y_true == c).sum()))
    return {"macro_f1": float(np.mean(f1s)),
            "balanced_acc": float(np.mean(recalls)),
            "ovr_auc": float(np.nanmean(aucs)),
            "per_class_auc": aucs, "counts": counts}


# ─────────────────────────────── probe 模型（torch）───────────────────────────────
def _train_probe(Xtr, ytr, Xte, n_classes, kind, epochs, device):
    import torch
    import torch.nn as nn
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6              # standardize：只用 train 統計（同時吸收各層尺度差）
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    d = Xtr.shape[1]
    if kind == "linear":
        model = nn.Linear(d, n_classes)
    else:
        h = max(64, d // 2)
        model = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, n_classes))
    model.to(device)
    freq = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    w = torch.tensor(len(ytr) / (n_classes * np.maximum(freq, 1)), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(model(Xtr_t), ytr_t).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(Xte_t), dim=1).cpu().numpy()
    return proba


def _group_kfold(pids, k, seed=0):
    """病人層級 K-fold：同一 pid 只落一折。回 [(train_idx, test_idx), ...]。"""
    uniq = sorted(set(pids))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    fold_of = {p: i % k for i, p in enumerate(uniq)}
    folds = np.array([fold_of[p] for p in pids])
    return [(np.where(folds != f)[0], np.where(folds == f)[0]) for f in range(k)]


def probe(args):
    files = sorted(glob.glob(os.path.join(args.cache, "*.npz")))
    if not files:
        print(f"⚠️ 快取空：{args.cache} → 先跑 --stage extract")
        return
    data = [np.load(f, allow_pickle=True) for f in files]
    y = np.array([int(d["label"]) for d in data])
    pids = [str(d["pid"]) for d in data]
    # binary 退路：稀有的 PCV/RAP 合併成 rest，穩定指標（快取仍是 3 類，這裡只是 probe 時壓縮）
    if args.binary:
        y = (y != 0).astype(np.int64)                       # nAMD=0 → 0；PCV/RAP → 1
        class_names = ["nAMD", "rest(PCV+RAP)"]
    else:
        class_names = CLASS_NAMES
    n_cls = len(class_names)
    kinds = ["linear", "mlp"] if args.probe == "both" else [args.probe]

    print(f"\n樣本={len(y)}  病人={len(set(pids))}  類別({'/'.join(class_names)})="
          f"{np.bincount(y, minlength=n_cls).tolist()}  folds={args.folds}\n")
    header = f"{'feature':<12}{'probe':<8}{'dim':>5}  {'macroF1':>8}{'bAcc':>8}{'ovrAUC':>8}"
    print(header + "   (mean±std over folds)")
    print("-" * len(header))

    dev = "cpu"
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    results = {}
    folds = _group_kfold(pids, args.folds, seed=args.seed)
    for feat in ALL_FEATS:
        if feat not in data[0].files:
            continue
        X = np.stack([d[feat] for d in data]).astype(np.float32)
        for kind in kinds:
            ms = {"macro_f1": [], "balanced_acc": [], "ovr_auc": []}
            for tr, te in folds:
                if len(np.unique(y[tr])) < 2 or len(te) == 0:
                    continue
                proba = _train_probe(X[tr], y[tr], X[te], n_cls, kind, args.epochs, dev)
                m = _metrics(y[te], proba.argmax(1), proba, n_cls)
                for kk in ms:
                    ms[kk].append(m[kk])
            agg = {kk: (float(np.mean(v)), float(np.std(v))) for kk, v in ms.items() if v}
            results[(feat, kind)] = agg
            if agg:
                print(f"{feat:<12}{kind:<8}{X.shape[1]:>5}  "
                      f"{agg['macro_f1'][0]:>6.3f}±{agg['macro_f1'][1]:.2f}"
                      f"{agg['balanced_acc'][0]:>6.3f}±{agg['balanced_acc'][1]:.2f}"
                      f"{agg['ovr_auc'][0]:>6.3f}±{agg['ovr_auc'][1]:.2f}")

    # ── 判讀摘要 ──
    print("\n── 判讀 ──")
    oct_scores = {f: results.get((f, kinds[0]), {}).get("ovr_auc", (float('nan'),))[0] for f in OCT_FEATS}
    oct_scores = {f: s for f, s in oct_scores.items() if s == s}
    if oct_scores:
        best_oct = max(oct_scores, key=oct_scores.get)
        print(f"Q1 OCTCube 疾病訊號最高層：{best_oct}  (ovrAUC={oct_scores[best_oct]:.3f})  "
              f"→ 峰值{'在' if best_oct == 'oct_b24' else '不在'} Block24")
    b0_best = max((results.get(('b0_latent', k), {}).get('ovr_auc', (float('nan'),))[0] for k in kinds),
                  default=float('nan'))
    if oct_scores and b0_best == b0_best:
        gap = oct_scores[best_oct] - b0_best
        verdict = ("瓶頸小/synergy 強 → 掛 B0 可行、偏好共訓(A)" if gap < 0.03 else
                   "有瓶頸 → 看 b0_level3 能否救；不行則獨立 semantic 學生(C)")
        print(f"Q2 b0_latent ovrAUC={b0_best:.3f} vs OCT 最佳 {oct_scores[best_oct]:.3f}  gap={gap:+.3f} → {verdict}")


def main():
    ap = argparse.ArgumentParser(description="語意保留 probe（Z1/Z2/Z3）")
    ap.add_argument("--stage", choices=["extract", "probe", "both"], default="both")
    # extract
    ap.add_argument("--h5-dir")
    ap.add_argument("--case-pooling")
    ap.add_argument("--b0-ckpt", default="probe_out/anisotropic_native_sqrtpct_seed43/best.pt")
    ap.add_argument("--b0-base-channels", type=int, default=16)
    ap.add_argument("--octcube-ckpt", default="ckpts/OCTCube.pth")
    ap.add_argument("--repo-dir", default="OCTCubeM-main/OCTCube")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all-visits", action="store_true", help="不做一眼一張去重（預設留最高 image_quality）")
    # shared / probe
    ap.add_argument("--cache", default="probe_out/semantics_probe_cache")
    ap.add_argument("--probe", choices=["linear", "mlp", "both"], default="both")
    ap.add_argument("--binary", action="store_true",
                    help="退路：三分類→nAMD vs rest(PCV+RAP)，PCV/RAP 太少時穩定指標")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.stage in ("extract", "both"):
        assert a.h5_dir and a.case_pooling, "--stage extract 需要 --h5-dir 與 --case-pooling"
        extract(a)
    if a.stage in ("probe", "both"):
        probe(a)


if __name__ == "__main__":
    main()
