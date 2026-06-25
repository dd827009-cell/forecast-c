"""Render a 4-panel verification image from a single training `.h5`.

Panels (clockwise from top-left):
    1. IR localizer with every B-scan's A-scan locus overlaid + fovea cross.
    2. Central B-scan with ILM (green) + RPE_BM (red) + BM_true (yellow).
    3. Thickness (RPE_BM - ILM) heatmap (D x W).
    4. 3D surface plot of ILM and RPE_BM.

Usage:
    python tools/verify_sample.py path/to/sample.h5 -o out.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def verify_sample(h5_path: str | Path, out_path: str | Path) -> Path:
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    h5_path = Path(h5_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        vol = h5["volume"][:]
        ir = h5["ir"][:]
        ilm = h5["ilm_y"][:]
        rpe = h5["rpe_bm_y"][:]
        bm_true = h5["bm_true_y"][:] if "bm_true_y" in h5 else None
        pos = h5["ascan_pos_ir"][:] if "ascan_pos_ir" in h5 else None
        valid = h5["valid_ascan_mask"][:]
        attrs = dict(h5.attrs)
        fx = float(attrs.get("fovea_ir_x", np.nan))
        fy = float(attrs.get("fovea_ir_y", np.nan))

    d, h_px, w_px = vol.shape
    central = d // 2

    fig = plt.figure(figsize=(14, 10))

    # Panel 1 — IR + B-scan lines
    ax1 = fig.add_subplot(2, 2, 1)
    if ir.size > 0:
        ax1.imshow(ir, cmap="gray", origin="upper")
    if pos is not None and pos.size > 0:
        for i in range(d):
            xs = pos[i, :, 0]
            ys = pos[i, :, 1]
            finite = np.isfinite(xs) & np.isfinite(ys)
            if finite.any():
                color = "red" if i == central else "lime"
                alpha = 1.0 if i == central else 0.35
                ax1.plot(xs[finite], ys[finite], color=color, linewidth=0.5,
                         alpha=alpha)
    if np.isfinite(fx) and np.isfinite(fy):
        ax1.plot(fx, fy, marker="+", color="cyan", markersize=18, mew=2)
    ax1.set_title(f"IR localizer  fovea=({fx:.1f},{fy:.1f})")
    ax1.set_xlabel("x px")
    ax1.set_ylabel("y px")

    # Panel 2 — central B-scan with layer overlay
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(vol[central], cmap="gray", origin="upper", aspect="auto")
    xs = np.arange(w_px)
    ax2.plot(xs, ilm[central], color="lime", linewidth=1.0, label="ILM")
    ax2.plot(xs, rpe[central], color="red", linewidth=1.0, label="RPE_BM")
    if bm_true is not None:
        ax2.plot(xs, bm_true[central], color="yellow", linewidth=1.0,
                 label="BM_true")
    ax2.set_title(f"Central B-scan (idx={central})")
    ax2.set_xlabel("A-scan")
    ax2.set_ylabel("depth px")
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3 — thickness heatmap
    ax3 = fig.add_subplot(2, 2, 3)
    thickness = np.where(valid, rpe - ilm, np.nan)
    im = ax3.imshow(thickness, cmap="viridis", aspect="auto", origin="upper")
    ax3.set_title("Thickness (RPE_BM - ILM) px")
    ax3.set_xlabel("A-scan")
    ax3.set_ylabel("B-scan")
    fig.colorbar(im, ax=ax3, fraction=0.035)

    # Panel 4 — 3D layer surface
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    step_w = max(1, w_px // 64)
    step_d = max(1, d // 32)
    Ws, Ds = np.meshgrid(np.arange(0, w_px, step_w), np.arange(0, d, step_d))
    ax4.plot_surface(
        Ws, Ds, ilm[::step_d, ::step_w], color="lime", alpha=0.5,
        linewidth=0, antialiased=False,
    )
    ax4.plot_surface(
        Ws, Ds, rpe[::step_d, ::step_w], color="red", alpha=0.5,
        linewidth=0, antialiased=False,
    )
    ax4.set_title("ILM (green) + RPE_BM (red)")
    ax4.set_xlabel("A-scan")
    ax4.set_ylabel("B-scan")
    ax4.set_zlabel("y px")
    ax4.invert_zaxis()

    title = (
        f"{attrs.get('patient_id','?')} / visit={attrs.get('visit_id','?')} "
        f"/ {attrs.get('laterality','?')} / IQ={float(attrs.get('image_quality', np.nan)):.1f} "
        f"/ flags='{attrs.get('flags','')}'"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render 4-panel verification PNG.")
    ap.add_argument("h5_path", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args(argv)

    path = verify_sample(args.h5_path, args.output)
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
