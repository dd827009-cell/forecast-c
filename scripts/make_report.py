"""產出 00000004.pat 的進度報告（圖片 + 文字摘要）。

Outputs to: report_00000004/
  bscans/        — 5 張 B-scan 切片 (含中央切片)
  overlays/      — 5 張 B-scan + ILM(綠)/BM(紅) 疊圖
  ir/            — IR localizer (含 25 條 B-scan 線 + fovea 十字)
  thickness/     — Thickness heatmap (RPE_BM - ILM)
  summary.txt    — 數據摘要
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\Administrator\Desktop\test1")
# Parser >= 2.0.0 writes to: h5_output/{sha1[:2]}/{sha1[2:4]}/{patient_id}/{visit_id}_{laterality}.h5
PATIENT_ID = "45611077"   # 00000004.pat's medical record number
_h = hashlib.sha1(PATIENT_ID.encode("utf-8")).hexdigest()
H5_DIR = ROOT / "h5_output" / _h[:2] / _h[2:4] / PATIENT_ID
H5_FILES = sorted(H5_DIR.glob("*.h5"))
OUT_ROOT = ROOT / "report_00000004"


def render_one(h5_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bscans").mkdir(exist_ok=True)
    (out_dir / "overlays").mkdir(exist_ok=True)
    (out_dir / "ir").mkdir(exist_ok=True)
    (out_dir / "thickness").mkdir(exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        vol = f["volume"][:]                # (D, H, W) float32 |fp16|
        ir = f["ir"][:]                     # (768, 768) uint8
        ilm = f["ilm_y"][:]                 # (D, W) float
        bm = f["rpe_bm_y"][:]               # (D, W) float
        pos = f["ascan_pos_ir"][:]          # (D, W, 2) float
        valid = f["valid_ascan_mask"][:]    # (D, W) bool
        iq_per = f["image_quality_per_bscan"][:]
        attrs = dict(f.attrs)

    D, H, W = vol.shape

    # Heidelberg display transform: sqrt(volume) then percentile-norm to [0,1].
    # Compute global percentile thresholds so all B-scans share the scale.
    disp = np.sqrt(np.clip(vol, 0, None))
    nz = disp[disp > 0]
    if nz.size:
        floor = np.percentile(nz, 15)
        disp[disp < floor] = 0
        valid_v = disp[disp > 0]
        p_lo = np.percentile(valid_v, 1) if valid_v.size else 0.0
        p_hi = np.percentile(valid_v, 99.9) if valid_v.size else 1.0
    else:
        p_lo, p_hi = 0.0, 1.0
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0
    disp = np.clip((disp - p_lo) / (p_hi - p_lo), 0, 1)
    side = attrs.get("laterality", "??")
    visit_id = attrs.get("visit_id", "")

    # ---- B-scan 切片：5 張 (頭、1/4、中、3/4、尾) ----
    picks = [0, D // 4, D // 2, (3 * D) // 4, D - 1]
    for idx in picks:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(disp[idx], cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"{visit_id} {side}  B-scan #{idx} / {D-1}   (quality={iq_per[idx]:.1f})")
        ax.set_xlabel("A-scan (W)")
        ax.set_ylabel("Depth (H, px)")
        fig.tight_layout()
        fig.savefig(out_dir / "bscans" / f"bscan_{idx:02d}.png", dpi=110)
        plt.close(fig)

    # ---- B-scan + ILM / BM 疊圖 ----
    for idx in picks:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(disp[idx], cmap="gray", aspect="auto", vmin=0, vmax=1)
        xs = np.arange(W)
        m = valid[idx]
        ilm_line = np.where(m & np.isfinite(ilm[idx]) & (ilm[idx] < 1e30), ilm[idx], np.nan)
        bm_line = np.where(m & np.isfinite(bm[idx]) & (bm[idx] < 1e30), bm[idx], np.nan)
        ax.plot(xs, ilm_line, color="#00ff66", lw=1.4, label="ILM")
        ax.plot(xs, bm_line, color="#ff3333", lw=1.4, label="BM (RPE_BM)")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"{visit_id} {side}  B-scan #{idx} with ILM + BM overlay")
        ax.set_xlabel("A-scan (W)")
        ax.set_ylabel("Depth (H, px)")
        fig.tight_layout()
        fig.savefig(out_dir / "overlays" / f"overlay_{idx:02d}.png", dpi=110)
        plt.close(fig)

    # ---- IR localizer + 25 條 B-scan 線 + fovea ----
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(ir, cmap="gray")
    for d in range(D):
        xs = pos[d, :, 0]; ys = pos[d, :, 1]
        m = valid[d] & np.isfinite(xs) & np.isfinite(ys)
        if m.any():
            ax.plot(xs[m], ys[m], color="#33aaff", lw=0.5, alpha=0.7)
    fx = float(attrs.get("fovea_ir_x", np.nan))
    fy = float(attrs.get("fovea_ir_y", np.nan))
    if np.isfinite(fx) and np.isfinite(fy):
        ax.plot([fx], [fy], marker="+", color="#ffcc00", markersize=18, mew=2,
                label=f"fovea ({fx:.1f},{fy:.1f})")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"{visit_id} {side}  IR localizer + {D} B-scan loci")
    ax.set_xlabel("IR x (px)"); ax.set_ylabel("IR y (px)")
    fig.tight_layout()
    fig.savefig(out_dir / "ir" / "ir_localizer.png", dpi=120)
    plt.close(fig)

    # ---- Thickness heatmap ----
    thick = bm - ilm
    thick = np.where(valid & np.isfinite(thick) & (np.abs(thick) < 1e30), thick, np.nan)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(thick, cmap="viridis", aspect="auto")
    cb = fig.colorbar(im, ax=ax); cb.set_label("BM - ILM (px)")
    ax.set_title(f"{visit_id} {side}  Retinal thickness (D x W)")
    ax.set_xlabel("A-scan (W)"); ax.set_ylabel("B-scan index (D)")
    fig.tight_layout()
    fig.savefig(out_dir / "thickness" / "thickness_heatmap.png", dpi=120)
    plt.close(fig)

    # ---- 文字摘要 ----
    lines = []
    lines.append(f"File: {h5_path.name}")
    lines.append(f"Path: {h5_path}")
    lines.append("")
    lines.append("== Datasets ==")
    lines.append(f"  volume            shape={vol.shape} dtype={vol.dtype}    # 3D OCT (D, H, W)")
    lines.append(f"  ir                shape={ir.shape} dtype={ir.dtype}    # IR / SLO localizer")
    lines.append(f"  ilm_y             shape={ilm.shape} dtype={ilm.dtype}  # ILM y 座標 (px)")
    lines.append(f"  rpe_bm_y              shape={bm.shape} dtype={bm.dtype}  # BM (RPE_BM) y 座標 (px)")
    lines.append(f"  ascan_pos_ir      shape={pos.shape} dtype={pos.dtype}  # 每根 A-scan 在 IR 上的 (x,y)")
    lines.append(f"  valid_ascan_mask  shape={valid.shape} dtype={valid.dtype}      # 該 A-scan 是否有效")
    lines.append(f"  image_quality_per_bscan shape={iq_per.shape}              # 每張 B-scan 品質分數")
    lines.append("")
    lines.append("== Attributes ==")
    for k, v in sorted(attrs.items()):
        s = str(v)
        if len(s) > 140: s = s[:140] + "..."
        lines.append(f"  {k} = {s}")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {h5_path.name} -> {out_dir}")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for h5 in H5_FILES:
        sub = OUT_ROOT / h5.stem  # 例如 20120612T023409_OD
        render_one(h5, sub)


if __name__ == "__main__":
    main()
