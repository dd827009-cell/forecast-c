"""Phase 2 縱向配對 dataset — 讀預計算 OCTCube latent 快取。取代 build_dataloader stub。

流程: 掃 h5 → 依 longitudinal_key(病歷號::眼) 分眼 → 同日去重(留 image_quality 高) → 按時間排序
     → 配對 (visit t → t+Δt) → 每對回 v_t/v_future(latent tokens) + Δt(年) + baseline(CST) + 未來厚度GT。
病人層級切分(防洩漏)。治療先 None(最小版 treatment-blind)，之後接 treatment.py/case_pooling.py。
"""
import glob
import os
import datetime
import collections
import random

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset, DataLoader

from forecast_c.data.oct_h5 import thickness_gt as _thickness_gt, cst as _cst


def latent_path_for(latent_dir, h5_dir, h5_path):
    """h5 路徑 → 對應快取 latent 路徑（與 precompute_latents._outpath 同規則）。"""
    rel = os.path.relpath(h5_path, h5_dir).replace(os.sep, "__").replace("/", "__")
    return os.path.join(latent_dir, rel[:-3] + ".pt")


def _parse_t(iso):
    try:
        return datetime.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except Exception:
        return None


def build_index(h5_dir, latent_dir):
    """掃 h5（只收有 latent 快取的）→ {longitudinal_key: [visit,...]}。"""
    groups = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(h5_dir, "**", "*.h5"), recursive=True)):
        lp = latent_path_for(latent_dir, h5_dir, f)
        if not os.path.exists(lp):
            continue
        try:
            with h5py.File(f, "r") as h:
                key = str(h.attrs.get("longitudinal_key", ""))
                t = str(h.attrs.get("acquisition_time_utc", ""))
                pid = str(h.attrs.get("patient_id", ""))
                q = (float(np.nanmedian(h["image_quality_per_bscan"][:]))
                     if "image_quality_per_bscan" in h else 0.0)
        except Exception:
            continue
        if key and t:
            groups[key].append({"h5": f, "latent": lp, "time": t, "day": t[:10], "q": q, "pid": pid})
    return groups


def make_pairs(groups, dedup=True, all_pairs=False):
    """{key:[visit]} → [(visit_t, visit_future), ...]（同日去重 + 排序 + 配對）。"""
    pairs = []
    for visits in groups.values():
        if dedup:
            byday = {}
            for v in visits:
                if v["day"] not in byday or v["q"] > byday[v["day"]]["q"]:
                    byday[v["day"]] = v
            visits = list(byday.values())
        visits = sorted(visits, key=lambda x: x["time"])
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


def split_pairs(pairs, val_frac=0.2, seed=0):
    """病人層級切分（防洩漏）：val 的病人所有配對不進 train。"""
    pats = sorted({p[0]["pid"] for p in pairs})
    rng = random.Random(seed); rng.shuffle(pats)
    val_pat = set(pats[:int(len(pats) * val_frac)])
    train = [p for p in pairs if p[0]["pid"] not in val_pat]
    val = [p for p in pairs if p[0]["pid"] in val_pat]
    return train, val


class PairedLatentDataset(Dataset):
    def __init__(self, pairs, out_h=25, out_w=512):
        self.pairs = pairs
        self.out_h, self.out_w = out_h, out_w

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        vt, vf = self.pairs[i]
        zt = torch.load(vt["latent"], map_location="cpu")["tokens"].float()   # (N,D)
        zf = torch.load(vf["latent"], map_location="cpu")["tokens"].float()
        tt, tf = _parse_t(vt["time"]), _parse_t(vf["time"])
        dt = ((tf - tt).days / 365.25) if (tt and tf) else 0.0
        c = _cst(vt["h5"]); c = c if (c == c) else 300.0
        th, mask = _thickness_gt(vf["h5"])                                    # (nb, W) 掃描寬度不一
        # resize 到固定 (out_h, out_w) → 不同掃描寬度(512/768…)才 stack 得起來
        th = F.interpolate(th[None, None].float(), size=(self.out_h, self.out_w),
                           mode="bilinear", align_corners=False)[0, 0]
        mask = F.interpolate(mask[None, None].float(), size=(self.out_h, self.out_w),
                             mode="nearest")[0, 0] > 0.5
        return {"v_t": zt, "v_future": zf,
                "dt": torch.tensor(float(dt), dtype=torch.float32),
                "baseline": torch.tensor(float((c - 300.0) / 100.0), dtype=torch.float32),
                "thickness_gt": th, "thickness_mask": mask, "treatment": None}


def collate_latent(samples):
    b = {k: torch.stack([s[k] for s in samples])
         for k in ("v_t", "v_future", "dt", "baseline", "thickness_gt", "thickness_mask")}
    b["treatment"] = None
    return b


def build_paired_latent_loader(cfg, latent_dir, h5_dir, split="train", batch_size=4,
                               val_frac=0.2, all_pairs=False, num_workers=4, seed=0):
    groups = build_index(h5_dir, latent_dir)
    pairs = make_pairs(groups, dedup=True, all_pairs=all_pairs)
    train, val = split_pairs(pairs, val_frac=val_frac, seed=seed)
    sel = train if split == "train" else val
    print(f"[paired_latent] {split}: {len(sel)} 對（train {len(train)}/val {len(val)}；眼組 {len(groups)}；"
          f"all_pairs={all_pairs}）")
    ds = PairedLatentDataset(sel, out_h=cfg.thickness.out_h, out_w=cfg.thickness.out_w)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                      collate_fn=collate_latent, num_workers=num_workers,
                      drop_last=(split == "train"))
