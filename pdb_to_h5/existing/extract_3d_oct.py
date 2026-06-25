"""
3D OCT Volume 提取與多格式輸出腳本
====================================
從 .eyepy 檔案提取 3D OCT 資料，輸出為多種格式，
以供 3D OCT Foundation Model 訓練/推論使用。

輸出格式：
  1. NumPy (.npy)      — 最通用，直接載入為 3D tensor
  2. NIfTI (.nii.gz)   — 醫學影像標準格式，保留 spacing header
  3. NPZ (.npz)        — 打包 volume + metadata + layers
  4. 3D 視覺化 (PNG)    — 類似 Fig1 的 3D 立體渲染圖

Usage:
  python extract_3d_oct.py
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from eyepy.core.eyevolume import EyeVolume


# ========================================================================
# 格式 1: NumPy (.npy) — Foundation Model 最常用
# ========================================================================

def export_npy(volume: EyeVolume, output_path: str, normalize: bool = True):
    """匯出為 .npy 格式的 3D numpy array。

    Args:
        volume: EyeVolume 物件
        output_path: 輸出路徑 (如 'output/OS_volume.npy')
        normalize: 若 True，正規化到 [0, 1] float32
    """
    data = volume.data.copy()  # shape: (D, H, W)

    if normalize:
        dmin, dmax = data.min(), data.max()
        if dmax > dmin:
            data = (data - dmin) / (dmax - dmin)
        data = data.astype(np.float32)

    np.save(output_path, data)
    print(f"[npy] Saved: {output_path}")
    print(f"       shape={data.shape}, dtype={data.dtype}, "
          f"range=[{data.min():.4f}, {data.max():.4f}]")
    return data


# ========================================================================
# 格式 2: NIfTI (.nii.gz) — 醫學影像標準，含空間 header
# ========================================================================

def export_nifti(volume: EyeVolume, output_path: str, normalize: bool = True):
    """匯出為 NIfTI 格式，保留 voxel spacing 資訊。

    NIfTI 是 3D OCT Foundation Model (如基於 MONAI 的模型) 最常見的輸入格式。
    Header 中包含 pixdim (voxel spacing) 讓模型知道實際物理尺寸。

    需要安裝: pip install nibabel
    """
    try:
        import nibabel as nib
    except ImportError:
        print("[nifti] ERROR: nibabel 未安裝。執行: pip install nibabel")
        return None

    data = volume.data.copy()

    if normalize:
        dmin, dmax = data.min(), data.max()
        if dmax > dmin:
            data = (data - dmin) / (dmax - dmin)
        data = data.astype(np.float32)

    # eyepy scale: (scale_z, scale_y, scale_x) in mm
    # volume.data shape: (n_bscans, bscan_height, bscan_width)
    #   axis 0 = B-scan index   → spacing = scale_z (inter-bscan)
    #   axis 1 = depth (axial)  → spacing = scale_y (axial)
    #   axis 2 = lateral        → spacing = scale_x (lateral)
    scale_z, scale_y, scale_x = volume.scale

    # 軸排列: (D, H, W) → (W, D, H) 使 3D Slicer 面板對應直覺切面:
    #   x axis (W=512, scale_x): 側向 → Sagittal 顯示 en-face 眼底俯視
    #   y axis (D=25,  scale_z): B-scan 方向 → Coronal 顯示 B-scan 視網膜層
    #   z axis (H=496, scale_y): 軸向深度 → Axial 顯示深度截面
    data_nifti = np.transpose(data, (2, 0, 1))  # (W, D, H)

    # NIfTI 仿射矩陣: 按新軸序 (scale_x, scale_z, scale_y)
    affine = np.diag([scale_x, scale_z, scale_y, 1.0])

    img = nib.Nifti1Image(data_nifti, affine)
    img.header.set_zooms((scale_x, scale_z, scale_y))
    img.header['xyzt_units'] = 2  # mm
    img.header['scl_slope'] = 1.0   # 明確禁用自動縮放，避免 NaN slope 問題
    img.header['scl_inter'] = 0.0
    img.header.set_qform(affine, code=1)  # 同步設定 qform，避免部分工具忽略方向

    nib.save(img, output_path)
    print(f"[nifti] Saved: {output_path}")
    print(f"        shape={data_nifti.shape} (W, D, H), "
          f"spacing=({scale_x:.5f}, {scale_z:.5f}, {scale_y:.5f}) mm")
    return data_nifti


# ========================================================================
# 格式 3: NPZ — 打包 volume + metadata + layers (一次性傳輸)
# ========================================================================

def export_npz(volume: EyeVolume, output_path: str, normalize: bool = True):
    """匯出為 .npz 壓縮包，包含 volume + scale + layers。

    適合需要同時存取影像和標註的場景。

    包含的 keys:
        'volume':    (D, H, W) float32
        'scale':     (3,) — [scale_z, scale_y, scale_x] in mm
        'localizer': (H, W) uint8 — SLO/IR 底圖
        + 每個 layer: 'layer_{name}' → (D, W) float32
    """
    data = volume.data.copy()

    if normalize:
        dmin, dmax = data.min(), data.max()
        if dmax > dmin:
            data = (data - dmin) / (dmax - dmin)
        data = data.astype(np.float32)

    save_dict = {
        'volume': data,
        'scale': np.array(volume.scale, dtype=np.float64),
        'localizer': volume.localizer.data,
    }

    # 加入層分割
    for name, layer in volume.layers.items():
        save_dict[f'layer_{name}'] = layer.data.astype(np.float32)

    np.savez_compressed(output_path, **save_dict)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[npz] Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"      keys: {list(save_dict.keys())}")
    return save_dict


# ========================================================================
# 格式 4: Metadata JSON — 空間資訊獨立檔案
# ========================================================================

def export_metadata_json(volume: EyeVolume, output_path: str, laterality: str = 'OS'):
    """匯出空間 metadata 為 JSON 檔。

    Foundation Model 可能需要讀取此資訊以正確處理各向異性 voxel。
    """
    scale_z, scale_y, scale_x = volume.scale
    d, h, w = volume.shape

    meta = {
        'laterality': laterality,
        'volume_shape': {'depth': d, 'height': h, 'width': w},
        'voxel_spacing_mm': {
            'inter_bscan': float(scale_z),
            'axial': float(scale_y),
            'lateral': float(scale_x),
        },
        'physical_size_mm': {
            'inter_bscan': float(d * scale_z),
            'axial': float(h * scale_y),
            'lateral': float(w * scale_x),
        },
        'layers_available': list(volume.layers.keys()),
        'data_range': {
            'min': float(volume.data.min()),
            'max': float(volume.data.max()),
        },
        'notes': {
            'axis_0': 'B-scan index (inferior→superior)',
            'axis_1': 'Axial depth (vitreous→choroid)',
            'axis_2': 'Lateral (nasal→temporal or vice versa)',
            'origin': 'eyepy convention: data[0]=inferior, data[-1]=superior',
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[json] Saved: {output_path}")
    return meta


# ========================================================================
# 3D 視覺化: 類似 Fig1_HTML.png 的立體渲染
# ========================================================================

def render_3d_oct(volume: EyeVolume, output_path: str, title: str = '3D OCT Volume'):
    """產生接近 OCT1.jpg 風格的 3D OCT 立體渲染圖。

    改進重點：
      - 顯式降取樣 MIP 至固定網格解析度，搭配 rstride=1/cstride=1 消除方塊感
      - 深度方向 (D=25) 上取樣 4× 以平滑側面與底面
      - ILM 表面同步降取樣，保持一致的平滑度
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import uniform_filter, zoom

    data = volume.data  # (D, H, W)  D=bscan index, H=depth, W=lateral
    d, h, w = data.shape
    scale_z, scale_y, scale_x = volume.scale

    # 正規化
    norm_data = data.astype(np.float32)
    dmin, dmax = norm_data.min(), norm_data.max()
    if dmax > dmin:
        norm_data = (norm_data - dmin) / (dmax - dmin)

    # --- 渲染參數 ---
    mesh_res = 200       # 主面網格解析度 (每邊 200 cells)
    depth_up = 4         # 深度方向上取樣倍率 (25→100)

    # --- Colormap ---
    oct_cmap = LinearSegmentedColormap.from_list(
        'oct_vol',
        [(0.00, '#000000'), (0.05, '#00111e'), (0.20, '#003d5c'),
         (0.42, '#0099b8'), (0.65, '#38c870'), (0.82, '#c8e820'),
         (1.00, '#ffffff')],
        N=256,
    )
    cap_cmap = LinearSegmentedColormap.from_list(
        'oct_cap',
        [(0, '#2a0400'), (0.25, '#7a1400'), (0.55, '#c84000'),
         (0.78, '#e87820'), (1.0, '#ffc060')],
        N=256,
    )

    # 物理尺寸
    ext_lat = w * scale_x    # X: lateral
    ext_dep = h * scale_y    # Z: axial depth
    ext_bsc = d * scale_z    # Y: inter-bscan

    # === MIP 投影 ===
    front_mip = np.max(norm_data, axis=0)  # (H, W) — 前視面
    side_mip  = np.max(norm_data, axis=2)  # (D, H) — 右側面
    bot_mip   = np.max(norm_data, axis=1)  # (D, W) — 底面

    # === 降取樣 + 色彩映射 (消除方塊感的關鍵) ===
    def ds_colormap(mip, target_h, target_w, alpha_gain=1.3, alpha_max=0.92):
        ds = zoom(mip, (target_h / mip.shape[0], target_w / mip.shape[1]), order=1)
        ds = np.clip(ds, 0, 1)
        rgba = oct_cmap(ds)
        rgba[..., 3] = np.clip(ds * alpha_gain, 0, alpha_max)
        return rgba

    front_col = ds_colormap(front_mip, mesh_res, mesh_res)

    sr_d = d * depth_up   # 25→100
    side_ds = zoom(side_mip, (sr_d / d, mesh_res / h), order=1)
    side_ds = np.clip(side_ds, 0, 1)
    side_plot = side_ds.T  # (mesh_res, sr_d) — 配合 meshgrid(ys, zs) 的 shape
    side_col = oct_cmap(side_plot)
    side_col[..., 3] = np.clip(side_plot * 1.3, 0, 0.92)

    bt_d = d * depth_up
    bot_col = ds_colormap(bot_mip, bt_d, mesh_res, alpha_gain=1.1, alpha_max=0.72)

    # === 建立 3D 圖 ===
    fig = plt.figure(figsize=(12, 9), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    # -- 前視面 Y=0 (lateral × depth) --
    xf = np.linspace(0, ext_lat, mesh_res)
    zf = np.linspace(0, ext_dep, mesh_res)
    Xf, Zf = np.meshgrid(xf, zf)
    Yf = np.zeros_like(Xf)
    ax.plot_surface(Xf, Yf, Zf, facecolors=front_col,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=2)

    # -- 右側面 X=ext_lat (bscan × depth) --
    ys = np.linspace(0, ext_bsc, sr_d)
    zs = np.linspace(0, ext_dep, mesh_res)
    Ys, Zs = np.meshgrid(ys, zs)   # shape (mesh_res, sr_d)
    Xs = np.full_like(Ys, ext_lat)
    ax.plot_surface(Xs, Ys, Zs, facecolors=side_col,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=2)

    # -- 底面 Z=ext_dep (bscan × lateral) --
    xb = np.linspace(0, ext_lat, mesh_res)
    yb = np.linspace(0, ext_bsc, bt_d)
    Xb, Yb = np.meshgrid(xb, yb)   # shape (bt_d, mesh_res)
    Zb = np.full_like(Xb, ext_dep)
    ax.plot_surface(Xb, Yb, Zb, facecolors=bot_col,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=1)

    # --- 橘紅 ILM 上表面 ---
    if 'ILM' in volume.layers:
        cap_raw = volume.layers['ILM'].data.astype(np.float32)
        nan_mask = np.isnan(cap_raw)
        if nan_mask.any():
            cap_raw = cap_raw.copy()
            cap_raw[nan_mask] = np.nanmedian(cap_raw)
        cap_px = uniform_filter(cap_raw, size=(3, 15))
    else:
        cap_px = np.argmax(norm_data > 0.25, axis=1).astype(np.float32)
        cap_px = uniform_filter(cap_px, size=(3, 15))

    cap_mm = cap_px * scale_y

    # 降取樣 ILM 表面至與其他面一致的解析度
    cap_d = d * depth_up
    cap_w = mesh_res
    cap_ds = zoom(cap_mm, (cap_d / d, cap_w / w), order=1)

    cap_min, cap_max = cap_ds.min(), cap_ds.max()
    cap_norm = np.clip((cap_ds - cap_min) / (cap_max - cap_min + 1e-6), 0, 1)
    cap_face = cap_cmap(cap_norm)

    # 假光照
    shading = np.clip(0.60 + 0.40 * cap_norm, 0.0, 1.0)
    cap_face[..., :3] *= shading[..., None]

    X_c = np.linspace(0, ext_lat, cap_w)
    Y_c = np.linspace(0, ext_bsc, cap_d)
    X_cg, Y_cg = np.meshgrid(X_c, Y_c)   # (cap_d, cap_w)

    ax.plot_surface(
        X_cg, Y_cg, cap_ds,
        facecolors=cap_face,
        rstride=1, cstride=1,
        linewidth=0, antialiased=True, shade=False,
        alpha=0.96, zorder=10,
    )

    # --- 黑底無軸 ---
    ax.set_xlim(0, ext_lat)
    ax.set_ylim(0, ext_bsc)
    ax.set_zlim(ext_dep, 0)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0, 0, 0, 1.0))
        axis.pane.set_edgecolor((0, 0, 0, 0))
        axis.line.set_color((0, 0, 0, 0))

    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=14)
    ax.view_init(elev=22, azim=-50)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    print(f"[3D render] Saved: {output_path}")
    plt.close()


# ========================================================================
# 格式 5: Fig1 風格 — 堆疊 B-scan 立體透視圖
# ========================================================================

def render_3d_bscan_stack(volume: EyeVolume, output_path: str,
                          title: str = '3D OCT Scan',
                          n_slices: int = None):
    """產生 Fig1_HTML.png 風格的堆疊 B-scan 3D 圖。

    將每張 B-scan 作為灰階切片疊放在 3D 空間中，
    並在表面繪製 ILM / BM 視網膜層曲線（白色）。
    附帶物理尺寸標註。

    Args:
        volume: EyeVolume 物件
        output_path: 輸出路徑
        title: 圖片標題
        n_slices: 顯示的切片數 (None=全部)
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    data = volume.data  # (D, H, W)
    d, h, w = data.shape
    scale_z, scale_y, scale_x = volume.scale

    # 正規化
    norm = data.astype(np.float32)
    dmin, dmax = norm.min(), norm.max()
    if dmax > dmin:
        norm = (norm - dmin) / (dmax - dmin)

    if n_slices is None:
        n_slices = d

    # 降取樣每張 B-scan 以保持效能 (每張 ~80×100 面)
    target_h, target_w = 80, 100

    fig = plt.figure(figsize=(14, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    # 選擇要顯示的切片
    if n_slices >= d:
        indices = list(range(d))
    else:
        indices = np.linspace(0, d - 1, n_slices, dtype=int).tolist()

    # 繪製每張 B-scan 切片
    x_coords = np.linspace(0, w * scale_x, target_w)
    z_coords = np.linspace(0, h * scale_y, target_h)
    X_mesh, Z_mesh = np.meshgrid(x_coords, z_coords)

    for i in indices:
        bscan = norm[i]
        bscan_ds = zoom(bscan, (target_h / h, target_w / w), order=1)
        bscan_ds = np.clip(bscan_ds, 0, 1)

        Y_mesh = np.full_like(X_mesh, i * scale_z)

        rgba = plt.cm.gray(bscan_ds)
        # 背景接近全透明，組織半透明
        rgba[..., 3] = np.clip(bscan_ds * 2.0, 0.03, 0.88)

        ax.plot_surface(X_mesh, Y_mesh, Z_mesh,
                        facecolors=rgba,
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)

    # 繪製層分割曲線 (ILM=白, BM=淺灰)
    layer_styles = [('ILM', 'white', 1.0), ('BM', '#bbbbbb', 0.8)]
    for layer_name, color, lw in layer_styles:
        if layer_name not in volume.layers:
            continue
        layer_data = volume.layers[layer_name].data.astype(np.float32)
        for i in indices:
            layer_row = layer_data[i]
            valid = ~np.isnan(layer_row)
            if valid.sum() < 10:
                continue
            lx = np.arange(w)[valid] * scale_x
            lz = layer_row[valid] * scale_y
            ly = np.full_like(lx, i * scale_z)
            ax.plot(lx, ly, lz, color=color, linewidth=lw, alpha=0.7)

    # 物理尺寸
    ext_lat = w * scale_x
    ext_bsc = d * scale_z
    ext_dep = h * scale_y

    ax.set_xlim(0, ext_lat)
    ax.set_ylim(0, ext_bsc)
    ax.set_zlim(ext_dep, 0)   # 深度向下

    # 尺寸標註 (類似 Fig1 的 mm 標籤)
    ax.set_xlabel(f'{ext_lat:.1f} mm', color='white', fontsize=11, labelpad=8)
    ax.set_ylabel(f'{ext_bsc:.1f} mm', color='white', fontsize=11, labelpad=8)
    ax.set_zlabel(f'{ext_dep:.1f} mm', color='white', fontsize=11, labelpad=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0, 0, 0, 1.0))
        axis.pane.set_edgecolor((0.3, 0.3, 0.3, 0.3))
        axis.line.set_color((0.3, 0.3, 0.3, 0.3))

    ax.tick_params(colors='white')
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=14)
    ax.view_init(elev=25, azim=-55)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    print(f"[3D stack] Saved: {output_path}")
    plt.close()


# ========================================================================
# B-scan Montage: 所有切面一覽
# ========================================================================

def render_bscan_montage(volume: EyeVolume, output_path: str, cols: int = 5):
    """將所有 B-scan 排列成 montage 圖，方便快速檢視。"""
    import matplotlib.pyplot as plt

    data = volume.data
    d, h, w = data.shape
    rows = (d + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
    axes = axes.flatten()

    for i in range(d):
        axes[i].imshow(data[i], cmap='gray', aspect='auto')
        axes[i].set_title(f'B-scan {i}', fontsize=8)
        axes[i].axis('off')

    # 隱藏空白子圖
    for i in range(d, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'All {d} B-scans', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[montage] Saved: {output_path}")
    plt.close()


# ========================================================================
# 主程式
# ========================================================================

def process_volume(eyepy_path: str, output_dir: str, eye: str):
    """處理單眼的 .eyepy 檔案，輸出所有格式。"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {eye} eye — {eyepy_path}")
    print(f"{'='*60}")

    volume = EyeVolume.load(eyepy_path)
    print(f"  Volume shape: {volume.shape}")
    print(f"  Scale (z, y, x): {volume.scale} mm")
    print(f"  Physical size: "
          f"{volume.shape[0]*volume.scale[0]:.2f} × "
          f"{volume.shape[1]*volume.scale[1]:.2f} × "
          f"{volume.shape[2]*volume.scale[2]:.2f} mm")
    print(f"  Layers: {list(volume.layers.keys())}")

    prefix = os.path.join(output_dir, f'{eye}')

    # 1. NumPy array
    export_npy(volume, f'{prefix}_volume.npy', normalize=True)

    # 2. NIfTI (if nibabel available)
    export_nifti(volume, f'{prefix}_volume.nii.gz', normalize=True)

    # 3. NPZ (volume + metadata + layers)
    export_npz(volume, f'{prefix}_volume_full.npz', normalize=True)

    # 4. Metadata JSON
    export_metadata_json(volume, f'{prefix}_metadata.json', laterality=eye)

    # 5. 3D render (OCT1.jpg 風格 — 彩色體積渲染)
    render_3d_oct(volume, f'{prefix}_3d_render.png', title=f'{eye} — 3D OCT Volume')

    # 6. 3D stacked B-scans (Fig1_HTML 風格 — 堆疊切片)
    render_3d_bscan_stack(volume, f'{prefix}_3d_stack.png',
                          title=f'{eye} — 3D OCT Scan')

    # 7. B-scan montage
    render_bscan_montage(volume, f'{prefix}_montage.png')

    return volume


if __name__ == '__main__':
    output_dir = 'output_3d_oct'

    # 處理左眼
    vol_os = process_volume(
        'topcon_OS_volume.eyepy', output_dir, 'OS')

    # 處理右眼
    vol_od = process_volume(
        'topcon_OD_volume.eyepy', output_dir, 'OD')

    # --- 摘要 ---
    print(f"\n{'='*60}")
    print("ALL OUTPUTS:")
    print(f"{'='*60}")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {f:40s}  {size_mb:8.2f} MB")

    print(f"\n--- Foundation Model 使用方式 ---")
    print("方法 A: 直接載入 numpy array")
    print("  vol = np.load('output_3d_oct/OS_volume.npy')  # (25, 496, 512) float32")
    print()
    print("方法 B: 載入 NIfTI (含 spacing)")
    print("  import nibabel as nib")
    print("  img = nib.load('output_3d_oct/OS_volume.nii.gz')")
    print("  vol = img.get_fdata()  # (512, 496, 25) — NIfTI axis order")
    print("  spacing = img.header.get_zooms()  # (0.01125, 0.00387, 0.24) mm")
    print()
    print("方法 C: 載入 NPZ (volume + layers)")
    print("  data = np.load('output_3d_oct/OS_volume_full.npz')")
    print("  vol = data['volume']       # (25, 496, 512)")
    print("  ilm = data['layer_ILM']    # (25, 512)")
    print("  bm  = data['layer_BM']     # (25, 512)")
    print("  scale = data['scale']      # [0.24, 0.00387, 0.01125]")
