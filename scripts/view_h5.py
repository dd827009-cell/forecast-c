#!/usr/bin/env python3
"""
view_h5.py — Interactive viewer for 3D OCT HDF5 files
======================================================

Usage:
  python view_h5.py path/to/file.h5
  python view_h5.py path/to/file.h5 --export-png    # 輸出靜態報告圖
  python view_h5.py path/to/folder                   # 批次查看資料夾內所有 .h5

Controls (interactive mode):
  ← →      Switch B-scan slice
  Home/End  Jump to first/last B-scan
  Q/Esc     Quit

Requirements:
  pip install h5py numpy matplotlib
"""

import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider


# ============================================================================
# Text report (always printed)
# ============================================================================

def print_report(h5_path: str):
    """Print a full text report of the HDF5 contents."""
    with h5py.File(h5_path, 'r') as f:
        attrs = dict(f.attrs)

        # Decode bytes
        for k, v in attrs.items():
            if isinstance(v, bytes):
                attrs[k] = v.decode()

        print(f"\n{'='*65}")
        print(f"  {os.path.basename(h5_path)}")
        print(f"{'='*65}")

        # Patient & visit
        print(f"\n  Patient ID:       {attrs.get('patient_id', '?')}")
        print(f"  Laterality:       {attrs.get('laterality', '?')}")
        print(f"  Visit date:       {attrs.get('visit_date', '?')}")
        print(f"  Acquisition:      {attrs.get('acquisition_time_utc', '?')}")

        # Volume
        vol = f['volume']
        D, H, W = vol.shape
        print(f"\n  Volume:           {D} × {H} × {W}  (D × H × W)")
        print(f"  Dtype:            {vol.dtype}")

        sz = attrs.get('scale_z_mm_per_bscan', 0)
        sy = attrs.get('scale_y_um_per_px', 0)
        sx = attrs.get('scale_x_mm_per_px', 0)
        print(f"  Physical size:    {D*sz:.2f} × {H*sy/1000:.2f} × {W*sx:.2f} mm")
        print(f"  Scale:            Z={sz:.4f} mm/B-scan, "
              f"Y={sy:.2f} µm/px, X={sx:.6f} mm/px")

        # Segmentation
        mask = f['valid_ascan_mask'][:]
        vc = int(mask.sum())
        vr = vc / mask.size * 100
        print(f"\n  Valid A-scans:    {vc}/{mask.size} ({vr:.1f}%)")
        print(f"  Seg layers:       {attrs.get('segmentation_types_available', '?')}")

        if vc > 0:
            ilm = f['ilm_y'][:]
            bm = f['bm_y'][:]
            thickness = bm[mask] - ilm[mask]
            t_um = thickness * sy
            print(f"  Thickness:        median={np.median(thickness):.1f} px "
                  f"({np.median(t_um):.0f} µm)")

        # Quality
        q = attrs.get('image_quality', 0)
        print(f"\n  Image quality:    {q:.1f}")
        if 'image_quality_per_bscan' in f:
            qpb = f['image_quality_per_bscan'][:]
            print(f"  Per-B-scan:       [{qpb.min():.1f}, {qpb.max():.1f}]")

        # IR
        ir = f['ir']
        has_ir = ir.shape[0] > 0
        print(f"\n  IR image:         {'Yes' if has_ir else 'No'}"
              f"{'  ' + str(ir.shape) if has_ir else ''}")

        # Fovea
        fx = attrs.get('fovea_ir_x', float('nan'))
        fy = attrs.get('fovea_ir_y', float('nan'))
        print(f"  Fovea (IR px):    ({fx:.1f}, {fy:.1f})"
              f"{'  ✓' if not np.isnan(fx) else '  ✗ undetected'}")

        # Line scans
        nls = attrs.get('n_line_scans', 0)
        print(f"\n  Line scans:       {nls}")
        if 'line_scans' in f and len(f['line_scans']) > 0:
            for k in f['line_scans']:
                ds = f['line_scans'][k]
                p = ds.attrs.get('pattern', b'?')
                if isinstance(p, bytes):
                    p = p.decode()
                print(f"    {k}: {ds.shape}, pattern={p}")

        # Keys
        print(f"\n  longitudinal_key: {attrs.get('longitudinal_key', '?')}")
        print(f"  visit_uid:        {attrs.get('visit_uid', '?')}")

        # Source
        print(f"\n  Parser version:   {attrs.get('parser_version', '?')}")
        print(f"  Flags:            {attrs.get('flags', '') or '(none)'}")
        sdb = attrs.get('source_sdb_path', '')
        print(f"  Source .sdb:      {os.path.basename(sdb) if sdb else '?'}")

        print(f"{'='*65}\n")


# ============================================================================
# Static PNG report
# ============================================================================

def export_png(h5_path: str, out_path: str = None):
    """Export a single-page visual summary as PNG."""
    if out_path is None:
        basename = os.path.basename(h5_path).replace('.h5', '_report.png')
        out_path = os.path.join(os.getcwd(), basename)

    with h5py.File(h5_path, 'r') as f:
        attrs = dict(f.attrs)
        for k, v in attrs.items():
            if isinstance(v, bytes):
                attrs[k] = v.decode()

        vol = f['volume'][:]
        ilm = f['ilm_y'][:]
        bm = f['bm_y'][:]
        mask = f['valid_ascan_mask'][:]
        D, H, W = vol.shape

        has_ir = f['ir'].shape[0] > 0
        ir = f['ir'][:] if has_ir else None

        has_ascan = 'ascan_pos_ir' in f and f['ascan_pos_ir'].shape[0] > 0
        ascan_pos = f['ascan_pos_ir'][:] if has_ascan else None

        qpb = None
        if 'image_quality_per_bscan' in f:
            qpb = f['image_quality_per_bscan'][:]

    fig = plt.figure(figsize=(18, 11), facecolor='#0e0e0e')
    fig.suptitle(
        f"{attrs.get('patient_id','?')}  |  {attrs.get('laterality','?')}  |  "
        f"{attrs.get('visit_date','?')[:10]}  |  "
        f"Quality: {attrs.get('image_quality',0):.1f}",
        color='white', fontsize=14, fontweight='bold', y=0.98
    )

    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.04, right=0.96, top=0.93, bottom=0.04)

    # --- Row 1: 4 representative B-scans with segmentation ---
    indices = [0, D // 4, D // 2, D - 1]
    for col, idx in enumerate(indices):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(vol[idx], cmap='gray', aspect='auto')

        # Overlay ILM (cyan) and BM (yellow)
        valid_w = np.where(mask[idx])[0]
        if len(valid_w) > 0:
            ax.plot(valid_w, ilm[idx, valid_w], color='#00d4ff', linewidth=0.6,
                    alpha=0.8)
            ax.plot(valid_w, bm[idx, valid_w], color='#ffd000', linewidth=0.6,
                    alpha=0.8)

        ax.set_title(f'B-scan {idx}/{D-1}', color='white', fontsize=9)
        ax.axis('off')

    # --- Row 2 left: IR with scan lines ---
    ax_ir = fig.add_subplot(gs[1, 0:2])
    if ir is not None:
        ax_ir.imshow(ir, cmap='gray', aspect='equal')
        # Draw B-scan positions
        if ascan_pos is not None:
            for d in range(0, D, max(1, D // 10)):
                xs = ascan_pos[d, :, 0]
                ys = ascan_pos[d, :, 1]
                color = '#00d4ff' if d == D // 2 else '#ffffff'
                alpha = 0.8 if d == D // 2 else 0.25
                lw = 1.0 if d == D // 2 else 0.4
                ax_ir.plot(xs, ys, color=color, linewidth=lw, alpha=alpha)

        # Fovea marker
        fx = attrs.get('fovea_ir_x', float('nan'))
        fy = attrs.get('fovea_ir_y', float('nan'))
        if not np.isnan(fx):
            ax_ir.plot(fx, fy, 'r+', markersize=12, markeredgewidth=2)

        ax_ir.set_title('IR en face + scan lines', color='white', fontsize=10)
    else:
        ax_ir.text(0.5, 0.5, 'No IR image', color='gray',
                   ha='center', va='center', transform=ax_ir.transAxes)
        ax_ir.set_title('IR en face', color='white', fontsize=10)
    ax_ir.axis('off')

    # --- Row 2 right: Thickness en face map ---
    ax_th = fig.add_subplot(gs[1, 2:4])
    thickness = np.full((D, W), np.nan, dtype=np.float32)
    for d in range(D):
        for w in range(W):
            if mask[d, w]:
                thickness[d, w] = bm[d, w] - ilm[d, w]

    sy = attrs.get('scale_y_um_per_px', 3.87)
    thickness_um = thickness * sy

    im = ax_th.imshow(thickness_um, cmap='RdYlGn_r', aspect='auto',
                      vmin=150, vmax=400, interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax_th, shrink=0.8, pad=0.02)
    cbar.set_label('µm', color='white', fontsize=9)
    cbar.ax.tick_params(colors='white', labelsize=8)
    ax_th.set_title('Retinal Thickness Map', color='white', fontsize=10)
    ax_th.set_xlabel('A-scan', color='gray', fontsize=8)
    ax_th.set_ylabel('B-scan', color='gray', fontsize=8)
    ax_th.tick_params(colors='gray', labelsize=7)

    # --- Row 3 left: En face projection ---
    ax_en = fig.add_subplot(gs[2, 0:2])
    # Mean projection at retinal layer depth
    enface = np.zeros((D, W), dtype=np.float32)
    for d in range(D):
        for w in range(W):
            if mask[d, w]:
                y0 = max(0, int(ilm[d, w]))
                y1 = min(H, int(bm[d, w]))
                if y1 > y0:
                    enface[d, w] = vol[d, y0:y1, w].mean()
    ax_en.imshow(enface, cmap='gray', aspect='auto')
    ax_en.set_title('En face (retinal slab)', color='white', fontsize=10)
    ax_en.axis('off')

    # --- Row 3 right: Quality per B-scan + metadata ---
    ax_q = fig.add_subplot(gs[2, 2])
    if qpb is not None:
        colors = ['#ff4444' if q < 15 else '#ffd000' if q < 25 else '#44cc44'
                  for q in qpb]
        ax_q.barh(range(D), qpb, color=colors, height=0.8)
        ax_q.set_xlabel('Quality', color='gray', fontsize=8)
        ax_q.set_ylabel('B-scan', color='gray', fontsize=8)
        ax_q.invert_yaxis()
        ax_q.axvline(x=15, color='#ff4444', linestyle='--', linewidth=0.5, alpha=0.5)
        ax_q.axvline(x=25, color='#ffd000', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_q.set_title('Quality / B-scan', color='white', fontsize=10)
    ax_q.set_facecolor('#1a1a1a')
    ax_q.tick_params(colors='gray', labelsize=7)
    ax_q.spines['bottom'].set_color('gray')
    ax_q.spines['left'].set_color('gray')
    ax_q.spines['top'].set_visible(False)
    ax_q.spines['right'].set_visible(False)

    # --- Row 3 far right: Metadata text ---
    ax_meta = fig.add_subplot(gs[2, 3])
    ax_meta.axis('off')
    valid_thick = thickness_um[np.isfinite(thickness_um)]

    meta_text = (
        f"Volume:  {D} × {H} × {W}\n"
        f"Size:    {D*attrs.get('scale_z_mm_per_bscan',0):.1f} × "
        f"{H*sy/1000:.1f} × {W*attrs.get('scale_x_mm_per_px',0):.1f} mm\n"
        f"\n"
        f"Valid:   {mask.sum()}/{mask.size} ({mask.mean()*100:.1f}%)\n"
        f"Thick:   {np.nanmedian(valid_thick):.0f} µm (median)\n"
        f"         [{np.nanmin(valid_thick):.0f}, {np.nanmax(valid_thick):.0f}] µm\n"
        f"\n"
        f"Seg:     {attrs.get('segmentation_types_available','?')}\n"
        f"IR:      {'Yes' if has_ir else 'No'}\n"
        f"Lines:   {attrs.get('n_line_scans', 0)}\n"
        f"Flags:   {attrs.get('flags','') or 'clean'}\n"
        f"\n"
        f"Parser:  v{attrs.get('parser_version','?')}\n"
        f"Key:     {attrs.get('longitudinal_key','?')}"
    )
    ax_meta.text(0.05, 0.95, meta_text, color='#cccccc', fontsize=8,
                 fontfamily='monospace', verticalalignment='top',
                 transform=ax_meta.transAxes)

    plt.savefig(out_path, dpi=150, facecolor='#0e0e0e', edgecolor='none',
                bbox_inches='tight')
    plt.close()
    print(f"  Report saved: {out_path}")


# ============================================================================
# Interactive viewer
# ============================================================================

def interactive_view(h5_path: str):
    """Launch interactive B-scan viewer with slider."""
    with h5py.File(h5_path, 'r') as f:
        attrs = dict(f.attrs)
        for k, v in attrs.items():
            if isinstance(v, bytes):
                attrs[k] = v.decode()

        vol = f['volume'][:]
        ilm = f['ilm_y'][:]
        bm = f['bm_y'][:]
        mask = f['valid_ascan_mask'][:]
        D, H, W = vol.shape

        has_ir = f['ir'].shape[0] > 0
        ir = f['ir'][:] if has_ir else None

        has_ascan = 'ascan_pos_ir' in f and f['ascan_pos_ir'].shape[0] > 0
        ascan_pos = f['ascan_pos_ir'][:] if has_ascan else None

    # Setup figure
    fig = plt.figure(figsize=(14, 7), facecolor='#111111')
    fig.suptitle(
        f"{attrs.get('patient_id','?')}  |  {attrs.get('laterality','?')}  |  "
        f"{attrs.get('visit_date','?')[:10]}",
        color='white', fontsize=12, fontweight='bold'
    )

    gs = GridSpec(2, 3, figure=fig, height_ratios=[10, 1],
                  hspace=0.15, wspace=0.25,
                  left=0.04, right=0.96, top=0.92, bottom=0.08)

    # B-scan panel (large)
    ax_bscan = fig.add_subplot(gs[0, 0:2])
    ax_bscan.set_facecolor('black')

    # IR panel
    ax_ir = fig.add_subplot(gs[0, 2])
    ax_ir.set_facecolor('black')

    if ir is not None:
        ax_ir.imshow(ir, cmap='gray', aspect='equal')
        ax_ir.set_title('IR en face', color='white', fontsize=9)
    else:
        ax_ir.text(0.5, 0.5, 'No IR', color='gray', ha='center', va='center',
                   transform=ax_ir.transAxes)
    ax_ir.axis('off')

    # Scan line on IR (will be updated)
    if ir is not None and ascan_pos is not None:
        # Draw all scan lines dimly
        for d in range(D):
            ax_ir.plot(ascan_pos[d, :, 0], ascan_pos[d, :, 1],
                       color='white', linewidth=0.3, alpha=0.15)

    ir_line, = ax_ir.plot([], [], color='#00d4ff', linewidth=1.5, alpha=0.9)

    # Slider
    ax_slider = fig.add_subplot(gs[1, :])
    slider = Slider(ax_slider, 'B-scan', 0, D - 1, valinit=D // 2,
                    valstep=1, color='#00d4ff')
    ax_slider.set_facecolor('#222222')

    # B-scan display elements
    bscan_img = ax_bscan.imshow(vol[D // 2], cmap='gray', aspect='auto')
    valid_w = np.where(mask[D // 2])[0]
    ilm_line, = ax_bscan.plot([], [], color='#00d4ff', linewidth=0.8, alpha=0.8,
                              label='ILM')
    bm_line, = ax_bscan.plot([], [], color='#ffd000', linewidth=0.8, alpha=0.8,
                             label='BM')
    ax_bscan.legend(loc='upper right', fontsize=8,
                    facecolor='#333333', edgecolor='gray', labelcolor='white')

    title_text = ax_bscan.set_title('', color='white', fontsize=10)

    def update(idx):
        idx = int(idx)
        bscan_img.set_data(vol[idx])

        valid_w = np.where(mask[idx])[0]
        if len(valid_w) > 0:
            ilm_line.set_data(valid_w, ilm[idx, valid_w])
            bm_line.set_data(valid_w, bm[idx, valid_w])
        else:
            ilm_line.set_data([], [])
            bm_line.set_data([], [])

        # Update title with per-bscan info
        thick = bm[idx, mask[idx]] - ilm[idx, mask[idx]] if mask[idx].sum() > 0 else []
        thick_str = f"  |  thickness={np.median(thick):.0f}px" if len(thick) > 0 else ""
        title_text.set_text(f'B-scan {idx}/{D-1}  |  '
                            f'valid={mask[idx].sum()}/{W}{thick_str}')

        # Update IR highlight line
        if ascan_pos is not None:
            ir_line.set_data(ascan_pos[idx, :, 0], ascan_pos[idx, :, 1])

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(D // 2)

    # Keyboard navigation
    def on_key(event):
        if event.key in ('right', 'up'):
            slider.set_val(min(slider.val + 1, D - 1))
        elif event.key in ('left', 'down'):
            slider.set_val(max(slider.val - 1, 0))
        elif event.key == 'home':
            slider.set_val(0)
        elif event.key == 'end':
            slider.set_val(D - 1)
        elif event.key in ('q', 'escape'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='View 3D OCT HDF5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python view_h5.py file.h5                # 互動式檢視 (可用方向鍵切換 B-scan)
  python view_h5.py file.h5 --export-png   # 輸出報告 PNG (存在當前目錄)
  python view_h5.py file.h5 --export-png -o D:/reports  # 指定輸出資料夾
  python view_h5.py folder/ --export-png   # 批次輸出資料夾內所有 .h5
  python view_h5.py file.h5 --text-only    # 只印文字報告
""")
    parser.add_argument('path', help='.h5 file or directory containing .h5 files')
    parser.add_argument('--export-png', action='store_true',
                        help='Export static report PNG instead of interactive view')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output directory for PNG reports (default: current directory)')
    parser.add_argument('--text-only', action='store_true',
                        help='Print text report only (no GUI)')
    args = parser.parse_args()

    # Collect .h5 files
    target = os.path.abspath(args.path)
    h5_files = []

    if os.path.isfile(target) and target.endswith('.h5'):
        h5_files.append(target)
    elif os.path.isdir(target):
        for root, dirs, files in os.walk(target):
            for fname in sorted(files):
                if fname.endswith('.h5'):
                    h5_files.append(os.path.join(root, fname))
    else:
        print(f"Error: {target} is not a .h5 file or directory")
        sys.exit(1)

    if not h5_files:
        print(f"No .h5 files found in {target}")
        sys.exit(1)

    print(f"Found {len(h5_files)} .h5 file(s)")

    for h5_path in h5_files:
        # Always print text report
        print_report(h5_path)

        if args.text_only:
            continue

        if args.export_png:
            # Compute output path
            png_name = os.path.basename(h5_path).replace('.h5', '_report.png')
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                out_path = os.path.join(args.output_dir, png_name)
            else:
                out_path = os.path.join(os.getcwd(), png_name)
            export_png(h5_path, out_path)
        elif len(h5_files) == 1:
            # Interactive only for single file
            interactive_view(h5_path)
        else:
            # Batch mode: auto export PNG
            png_name = os.path.basename(h5_path).replace('.h5', '_report.png')
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                out_path = os.path.join(args.output_dir, png_name)
            else:
                out_path = os.path.join(os.getcwd(), png_name)
            export_png(h5_path, out_path)

    if args.export_png or len(h5_files) > 1:
        out_loc = args.output_dir or os.getcwd()
        print(f"\nDone! {len(h5_files)} report(s) exported to: {out_loc}")


if __name__ == '__main__':
    main()