"""預計算凍結 OCTCube latent → 快取（每 h5 一份 fp16 tokens+cls）。

Phase 2 最小版：encoder 凍結，同一 h5 會出現在多個配對 → 先算一次快取，訓練直接讀，省大量重複 forward。

用法（先小批測）:
  python -m forecast_c.train.precompute_latents --h5-dir /mnt/d/h5out --out /mnt/d/latents --limit 30
全量:
  python -m forecast_c.train.precompute_latents --h5-dir /mnt/d/h5out --out /mnt/d/latents --workers-note

每 h5 latent = 5120×1024 → fp16 約 10.5MB；8129 個 ≈ 85GB。可續跑（已存的跳過）。
"""
import argparse
import glob
import os
import time

import h5py
import torch

from forecast_c.model.encoder import build_octcube_encoder, OCTCubeTokenEncoder


def _outpath(out_dir, h5_dir, f):
    rel = os.path.relpath(f, h5_dir).replace(os.sep, "__").replace("/", "__")[:-3] + ".pt"
    return os.path.join(out_dir, rel)


def main():
    ap = argparse.ArgumentParser(description="預計算 OCTCube latent 快取")
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="ckpts/OCTCube.pth")
    ap.add_argument("--repo-dir", default="OCTCubeM-main/OCTCube")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="只做前 N 個(0=全部)")
    a = ap.parse_args()

    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"device = {dev}")
    print("載入凍結 OCTCube ...")
    enc = build_octcube_encoder(ckpt_path=a.ckpt, repo_dir=a.repo_dir, device=dev)

    files = sorted(glob.glob(os.path.join(a.h5_dir, "**", "*.h5"), recursive=True))
    if a.limit:
        files = files[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    print(f"共 {len(files)} 個 h5 → {a.out}")

    done = skipped = 0
    t0 = time.time()
    for i, f in enumerate(files):
        outp = _outpath(a.out, a.h5_dir, f)
        if os.path.exists(outp):
            skipped += 1
            continue
        try:
            with h5py.File(f, "r") as h:
                vol = h["volume"][:]
            x = OCTCubeTokenEncoder.prep(vol).to(dev)
            with torch.no_grad():
                tokens, cls = enc(x)
            torch.save({"tokens": tokens[0].half().cpu().contiguous(),
                        "cls": cls[0].half().cpu().contiguous()}, outp)
            done += 1
        except Exception as e:
            print(f"  [FAIL] {f}: {e}")
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            dt = time.time() - t0
            rate = (done + skipped) / max(dt, 1e-6)
            eta = (len(files) - i - 1) / max(rate, 1e-6)
            print(f"  {i+1}/{len(files)}  new={done} skip={skipped}  {rate:.1f}/s  ETA {eta/60:.1f}min")
    print(f"完成: 新算 {done}、跳過 {skipped}、耗時 {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
