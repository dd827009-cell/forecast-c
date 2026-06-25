"""
Topcon OCT → eyepy EyeVolume 轉接器 (Adapter)
==============================================
將 Topcon .sdb/.pdb/.edb 逆向工程後萃取的資料，
組裝成標準的 eyepy.EyeVolume 3D 物件。

★ 設計理念：「繞過 eyepy 原生 reader」★
  接受記憶體中的 Numpy 陣列 + CSV metadata，
  手動實例化 EyeVolume / EyeEnface / EyeBscanMeta 等物件。

座標轉換策略：
  Topcon 使用視角度 (°) 座標系，原點在 fovea (0,0)。
  eyepy 的 EyeBscanMeta 使用物理座標 (mm)，原點在 SLO 圖像左上角。
  轉換公式 (Gullstrand 眼球模型 1° ≈ 0.288 mm)：
    x_mm = x_deg × 0.288 + FOV_mm / 2
    y_mm = FOV_mm / 2 − y_deg × 0.288  （y 軸翻轉：superior↑ → 圖像 row↓）

B-scan 排列順序：
  eyepy 約定 data[0] = 最底部 B-scan, data[-1] = 最頂部 B-scan。
  Topcon scanIndex 0 = y=+10° (superior, 頂部), scanIndex 24 = y=-10° (inferior, 底部)。
  因此需要反轉 (flip) volume data、bscan_meta、layer_heights 的 z 軸順序。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

# ========================================================================
# 常數定義
# ========================================================================

# Gullstrand 眼球模型：1 視角度 ≈ 0.288 mm
DEG_TO_MM = 0.288

# Topcon SLO 視野角度
SLO_FOV_DEG = 30.0

# 由 FOV 推算：30° × 0.288 = 8.64 mm
SLO_FOV_MM = SLO_FOV_DEG * DEG_TO_MM  # 8.64 mm

# SLO 像素尺寸 (768 px for 30° FOV)
SLO_PIXELS = 768

# SLO mm/pixel
SLO_SCALE = SLO_FOV_MM / SLO_PIXELS  # ≈ 0.01125 mm/pixel


# ========================================================================
# .sdb 二進位檔讀取函數
# ========================================================================

def read_sdb_images(
    sdb_path: str,
    image_details_csv: str,
    series_id: Optional[int] = None,
) -> dict:
    """從 .sdb 二進位檔案直接讀取 uint16 OCT + uint8 SLO 像素資料。

    利用 image_details CSV 中的 pixel_offset_in_file 欄位，
    精確 seek 到每張影像的起始位元組，讀取完整 16-bit 深度的原始資料。

    與 BMP 匯出 (8-bit) 相比，保留了完整的 16-bit 動態範圍 (0–65535)，
    對量化分析 (層厚度、反射率) 更為精確。

    Args:
        sdb_path: .sdb 檔案的完整路徑
            (如 'D:/topcon_data/00000036.sdb')
        image_details_csv: image_details CSV 的路徑
            (如 'test/18_image_details.csv')
        series_id: 指定要讀取的 series ID。若 None，會從 sdb 檔名自動推斷
            (例如 '00000036.sdb' → series_id=36)。

    Returns:
        dict 包含以下鍵值：
            'volume_data': np.ndarray, shape (N_bscans, H, W), dtype uint16
                           B-scan 像素陣列，Topcon 原始順序 (top-first)
            'slo_data':    np.ndarray, shape (H, W), dtype uint8
                           SLO/IR 紅外底圖
            'series_id':   int, 使用的 series ID
            'image_ids':   list[int], 各 B-scan 的 imageID
            'n_bscans':    int, B-scan 數量
            'bscan_size':  tuple (height, width)
            'slo_size':    tuple (height, width)
            'pixel_fmt':   str, B-scan 像素格式 ('uint16')
            'source_file': str, .sdb 檔名

    Raises:
        FileNotFoundError: 找不到 .sdb 或 CSV
        ValueError: CSV 中無符合條件的資料
    """
    sdb_path = Path(sdb_path)
    if not sdb_path.exists():
        raise FileNotFoundError(f"找不到 .sdb 檔案: {sdb_path}")

    df = pd.read_csv(image_details_csv)

    # 自動推斷 series_id
    if series_id is None:
        sdb_name = sdb_path.name  # e.g. '00000036.sdb'
        matching = df[df['sourceFile'] == sdb_name]
        if len(matching) == 0:
            raise ValueError(f"CSV 中找不到 sourceFile='{sdb_name}'")
        series_id = matching['seriesID'].iloc[0]
        print(f"[INFO] 自動從檔名推斷 series_id={series_id}")

    # 篩選此 series 的主影像 (typeHex=0x40000000)
    series_df = df[(df['seriesID'] == series_id) & (df['typeHex'] == '0x40000000')]
    if len(series_df) == 0:
        raise ValueError(f"CSV 中找不到 seriesID={series_id} 的主影像")

    slo_rows = series_df[series_df['imageRole'] == 'SLO']
    bscan_rows = series_df[series_df['imageRole'] == 'BScan'].sort_values('imageID')

    if len(slo_rows) == 0:
        raise ValueError(f"Series {series_id} 中找不到 SLO")
    if len(bscan_rows) == 0:
        raise ValueError(f"Series {series_id} 中找不到 BScan")

    slo_info = slo_rows.iloc[0]
    n_bscans = len(bscan_rows)

    # 讀取 .sdb 二進位檔
    with open(sdb_path, 'rb') as f:
        # --- 讀取 SLO (uint8) ---
        slo_h = int(slo_info['breite_height'])
        slo_w = int(slo_info['hoehe_width'])
        slo_offset = int(slo_info['pixel_offset_in_file'])
        slo_bytes = slo_h * slo_w  # uint8, 1 byte/pixel

        f.seek(slo_offset)
        slo_raw = f.read(slo_bytes)
        if len(slo_raw) != slo_bytes:
            raise IOError(f"SLO 讀取不足: 預期 {slo_bytes} bytes, 實際 {len(slo_raw)}")
        slo_data = np.frombuffer(slo_raw, dtype=np.uint8).reshape(slo_h, slo_w)

        # --- 讀取 BScans (uint16) ---
        b0 = bscan_rows.iloc[0]
        bscan_h = int(b0['breite_height'])   # 496 (depth)
        bscan_w = int(b0['hoehe_width'])     # 512 (lateral)
        bpp = int(b0['bpp'])                  # 2
        pixel_bytes = int(b0['pixel_bytes'])  # 507904

        dtype = np.uint16 if b0['pixel_fmt'] == 'uint16' else np.uint8

        volume_data = np.empty((n_bscans, bscan_h, bscan_w), dtype=dtype)
        image_ids = []

        for i, (_, row) in enumerate(bscan_rows.iterrows()):
            offset = int(row['pixel_offset_in_file'])
            f.seek(offset)
            raw = f.read(pixel_bytes)
            if len(raw) != pixel_bytes:
                raise IOError(
                    f"BScan {i} (imageID={row['imageID']}) 讀取不足: "
                    f"預期 {pixel_bytes} bytes, 實際 {len(raw)}")
            volume_data[i] = np.frombuffer(raw, dtype=dtype).reshape(bscan_h, bscan_w)
            image_ids.append(int(row['imageID']))

    result = {
        'volume_data': volume_data,
        'slo_data': slo_data,
        'series_id': series_id,
        'image_ids': image_ids,
        'n_bscans': n_bscans,
        'bscan_size': (bscan_h, bscan_w),
        'slo_size': (slo_h, slo_w),
        'pixel_fmt': str(b0['pixel_fmt']),
        'source_file': str(b0['sourceFile']),
    }

    print(f"[OK] 已從 {sdb_path.name} 讀取 Series {series_id}:")
    print(f"  SLO:    {slo_data.shape} {slo_data.dtype} (offset={slo_offset})")
    print(f"  BScans: {volume_data.shape} {volume_data.dtype} "
          f"(range=[{volume_data.min()}, {volume_data.max()}])")

    return result


def read_sdb_all_series(
    sdb_dir: str,
    image_details_csv: str,
) -> dict[int, dict]:
    """讀取目錄中所有 .sdb 檔案的影像資料。

    Args:
        sdb_dir: 包含 .sdb 檔案的目錄路徑
        image_details_csv: image_details CSV 路徑

    Returns:
        dict: {series_id: read_sdb_images() 的回傳結果}
    """
    df = pd.read_csv(image_details_csv)
    sdb_files = sorted(Path(sdb_dir).glob('*.sdb'))

    if not sdb_files:
        raise FileNotFoundError(f"在 {sdb_dir} 中找不到 .sdb 檔案")

    results = {}
    for sdb_path in sdb_files:
        sdb_name = sdb_path.name
        matching = df[df['sourceFile'] == sdb_name]
        if len(matching) == 0:
            print(f"[WARN] CSV 中找不到 {sdb_name}，跳過")
            continue
        sid = int(matching['seriesID'].iloc[0])
        try:
            results[sid] = read_sdb_images(sdb_path, image_details_csv, sid)
        except Exception as e:
            print(f"[ERROR] 讀取 {sdb_name} (Series {sid}) 失敗: {e}")

    return results


def read_bmp_volume(
    bmp_dir: str,
    oct_pattern: str = 'oct_{i:04d}_512x496.bmp',
    ir_pattern: str = 'ir_{i:04d}_768x768.bmp',
    n_bscans: int = 25,
) -> dict:
    """從 BMP 匯出目錄讀取 OCT 體積 + IR 底圖。

    適用於使用者已將二進位資料匯出為 BMP 的情況。
    注意：BMP 為 8-bit，動態範圍低於原始 .sdb 的 uint16。

    Args:
        bmp_dir: BMP 檔案所在目錄
        oct_pattern: OCT BMP 檔名模式 (需含 {i} 佔位符)
        ir_pattern: IR BMP 檔名模式 (需含 {i} 佔位符)
        n_bscans: OCT B-scan 的數量

    Returns:
        dict 包含以下鍵值：
            'volume_data': np.ndarray, shape (N, H, W), dtype uint8
            'slo_data':    np.ndarray, shape (H, W), dtype uint8
    """
    bmp_dir = Path(bmp_dir)

    # 讀取 OCT BScans
    oct_images = []
    for i in range(n_bscans):
        fname = oct_pattern.format(i=i)
        fpath = bmp_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"找不到 OCT BMP: {fpath}")
        img = np.array(Image.open(fpath).convert('L'))
        oct_images.append(img)
    volume_data = np.stack(oct_images, axis=0)

    # 讀取 IR
    ir_fname = ir_pattern.format(i=0)
    ir_path = bmp_dir / ir_fname
    if not ir_path.exists():
        raise FileNotFoundError(f"找不到 IR BMP: {ir_path}")
    slo_data = np.array(Image.open(ir_path).convert('L'))

    print(f"[OK] 已從 BMP 目錄讀取:")
    print(f"  OCT: {volume_data.shape} {volume_data.dtype} "
          f"(range=[{volume_data.min()}, {volume_data.max()}])")
    print(f"  IR:  {slo_data.shape} {slo_data.dtype}")

    return {
        'volume_data': volume_data,
        'slo_data': slo_data,
    }


def read_bmp_by_series(
    parent_dir: str,
    series_id: int,
    csv_dir: str,
    oct_pattern: str = 'oct_{i:04d}_{w}x{h}.bmp',
    ir_pattern: str = 'ir_{i:04d}_{w}x{h}.bmp',
) -> dict:
    """從 series ID 對應的子資料夾讀取 BMP 影像。

    資料夾結構：
        parent_dir/
            00000036/   ← series_id=36, 包含 25張 OCT + 1張 IR
            00000040/   ← series_id=40, 包含 25張 OCT + 1張 IR
            ...

    會自動從 06_bscan_metadata.csv 讀取該 series 的 B-scan 數量和影像尺寸，
    並據此定位正確的 BMP 檔案。

    Args:
        parent_dir: 包含所有 series 子資料夾的父目錄
        series_id: 目標 series ID (如 36=OS, 40=OD)
        csv_dir: CSV 目錄 (用於讀取 metadata)
        oct_pattern: OCT BMP 檔名模式。支援 {i}, {w}, {h} 佔位符
        ir_pattern: IR BMP 檔名模式。支援 {i}, {w}, {h} 佔位符

    Returns:
        dict: {'volume_data': ndarray, 'slo_data': ndarray}
    """
    # 從 CSV 取得該 series 的 B-scan 數量和尺寸
    meta_csv = os.path.join(csv_dir, '06_bscan_metadata.csv')
    meta_df = pd.read_csv(meta_csv)
    series_meta = meta_df[meta_df['seriesID'] == series_id]
    if len(series_meta) == 0:
        raise ValueError(f"CSV 中找不到 seriesID={series_id}")

    n_bscans = len(series_meta)
    # Topcon: imgSizeX = depth (496), imgSizeY = width (512)
    bscan_height = int(series_meta.iloc[0]['imgSizeX'])
    bscan_width = int(series_meta.iloc[0]['imgSizeY'])

    # 子資料夾名稱: 零填充 8 位數字
    folder_name = f'{series_id:08d}'
    series_dir = Path(parent_dir) / folder_name
    if not series_dir.exists():
        raise FileNotFoundError(
            f"找不到 series 資料夾: {series_dir}")

    print(f"[BMP] Series {series_id}: "
          f"資料夾={folder_name}, "
          f"{n_bscans} B-scans, "
          f"尺寸={bscan_width}x{bscan_height}")

    # 自動偵測 BMP 檔案命名模式
    # 嘗試兩種常見模式
    bmp_files = sorted(series_dir.glob('oct_*.bmp'))
    ir_files = sorted(series_dir.glob('ir_*.bmp'))

    if len(bmp_files) >= n_bscans:
        # 直接用找到的檔案
        oct_images = []
        for fpath in bmp_files[:n_bscans]:
            img = np.array(Image.open(fpath).convert('L'))
            oct_images.append(img)
        volume_data = np.stack(oct_images, axis=0)
    else:
        # 用 pattern 嘗試
        resolved_oct = oct_pattern.replace('{w}', str(bscan_width)).replace('{h}', str(bscan_height))
        volume_data = _read_bmp_sequence(series_dir, resolved_oct, n_bscans, 'OCT')

    if len(ir_files) >= 1:
        slo_data = np.array(Image.open(ir_files[0]).convert('L'))
    else:
        resolved_ir = ir_pattern.replace('{w}', '768').replace('{h}', '768')
        ir_fname = resolved_ir.format(i=0)
        ir_path = series_dir / ir_fname
        if not ir_path.exists():
            raise FileNotFoundError(f"找不到 IR BMP: {ir_path}")
        slo_data = np.array(Image.open(ir_path).convert('L'))

    print(f"  OCT: {volume_data.shape} {volume_data.dtype} "
          f"(range=[{volume_data.min()}, {volume_data.max()}])")
    print(f"  IR:  {slo_data.shape} {slo_data.dtype}")

    return {
        'volume_data': volume_data,
        'slo_data': slo_data,
    }


def _read_bmp_sequence(directory: Path, pattern: str, count: int, label: str) -> np.ndarray:
    """讀取一系列命名規則一致的 BMP 檔案。"""
    images = []
    for i in range(count):
        fname = pattern.format(i=i)
        fpath = directory / fname
        if not fpath.exists():
            raise FileNotFoundError(f"找不到 {label} BMP: {fpath}")
        images.append(np.array(Image.open(fpath).convert('L')))
    return np.stack(images, axis=0)


def find_volume_series(csv_dir: str, min_bscans: int = 10) -> list[dict]:
    """從 CSV 自動識別哪些 series 是 Volume scan（多張 B-scan）。

    Args:
        csv_dir: CSV 目錄
        min_bscans: 最少 B-scan 數量才算 Volume scan

    Returns:
        list of dict: [{'series_id': 36, 'n_bscans': 25, 'width': 512, 'height': 496}, ...]
    """
    meta_csv = os.path.join(csv_dir, '06_bscan_metadata.csv')
    meta_df = pd.read_csv(meta_csv)

    result = []
    for sid, grp in meta_df.groupby('seriesID'):
        if len(grp) >= min_bscans:
            result.append({
                'series_id': int(sid),
                'n_bscans': len(grp),
                'width': int(grp.iloc[0]['imgSizeY']),
                'height': int(grp.iloc[0]['imgSizeX']),
            })
    return result


# ========================================================================
# CSV 載入函數
# ========================================================================

def load_bscan_positions(csv_path: str, series_id: int) -> pd.DataFrame:
    """讀取 B-scan 空間座標 CSV (07_bscans_positions.csv)。

    Returns:
        按 scanIndex 排序的 DataFrame
    """
    df = pd.read_csv(csv_path)
    sub = df[df['seriesID'] == series_id].sort_values('scanIndex').reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(f"CSV 中找不到 seriesID={series_id}")
    return sub


def load_bscan_metadata_csv(csv_path: str, series_id: int) -> pd.DataFrame:
    """讀取 B-scan 元資料 CSV (06_bscan_metadata.csv)。"""
    df = pd.read_csv(csv_path)
    sub = df[df['seriesID'] == series_id].reset_index(drop=True)
    return sub


def load_layer_boundaries(csv_path: str, series_id: int) -> dict[str, np.ndarray]:
    """讀取分層邊界 CSV (12_layer_boundaries.csv)。

    Returns:
        dict: layerName → shape (n_bscans, width) 的高度圖，已反轉 z 軸
              NaN 表示該位置無效
    """
    df = pd.read_csv(csv_path)
    sub = df[df['seriesID'] == series_id]

    layers = {}
    for layer_name in sub['layerName'].unique():
        layer_df = sub[sub['layerName'] == layer_name].sort_values('imageID').reset_index(drop=True)

        # 檢查是否全部無效 (如 Layer7 全為 NaN)
        if layer_df['validAscans'].sum() == 0:
            continue

        height_maps = []
        for _, row in layer_df.iterrows():
            boundary_str = row['boundaryData']
            # 分號分隔的浮點數，'NaN' 字串保持為 np.nan
            values = np.array([
                np.nan if v.strip() == 'NaN' else float(v.strip())
                for v in boundary_str.split(';')
            ], dtype=np.float32)
            height_maps.append(values)

        height_array = np.stack(height_maps, axis=0)  # (n_bscans, width)
        # 反轉 z 軸: 與 volume data 保持一致
        # layer[0] = inferior (scanIndex 24), layer[-1] = superior (scanIndex 0)
        height_array = np.flip(height_array, axis=0).copy()
        layers[layer_name] = height_array

    return layers


def load_slo_image(image_dir: str, series_id: int) -> np.ndarray:
    """載入 SLO (IR localizer) 圖片。

    Args:
        image_dir: 圖檔目錄路徑 (如 csv_output/ir_images)
        series_id: series ID

    Returns:
        shape (H, W), dtype uint8
    """
    fname = f"series_{series_id}_SLO_raw.png"
    fpath = os.path.join(image_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"找不到 SLO 圖檔: {fpath}")
    return np.array(Image.open(fpath).convert('L'))


# ========================================================================
# 核心 Adapter：從記憶體陣列組裝 EyeVolume
# ========================================================================

def build_eyevolume_from_arrays(
    volume_data: np.ndarray,
    ir_localizer: np.ndarray,
    geometry_coords: list[dict],
    scale_meta: dict,
    layer_heights: Optional[dict[str, list[np.ndarray]]] = None,
    laterality: str = 'OS',
    axial_length_mm: Optional[float] = None,
    topcon_scan_order_top_first: bool = True,
) -> 'EyeVolume':
    """★ 核心函數：從記憶體中的 Numpy 陣列手動組裝 eyepy.EyeVolume。

    這是「繞過 eyepy 原生 reader」的轉接器，直接用你記憶體中的資料
    實例化標準的 EyeVolume 3D 物件。

    Args:
        volume_data: shape (N_bscans, Z, X) 的 OCT 體積資料，dtype uint8 或 float32。
            ★ 注意：如果 topcon_scan_order_top_first=True，
              volume_data[0] 是最頂部 (superior) 的 B-scan，
              函數內部會自動反轉為 eyepy 的 bottom-first 順序。
        ir_localizer: shape (H, W) 的 SLO/IR 底圖，dtype uint8。
        geometry_coords: 長度 N_bscans 的列表，每個元素是字典：
            {"start_x": float, "start_y": float, "end_x": float, "end_y": float}
            ★ 座標單位：度 (°)，以 fovea 為原點，y+ = superior。
            ★ 順序必須與 volume_data 的 z 軸對齊。
        scale_meta: 物理比例字典：
            {"scale_x": float,  # B-scan 橫向 mm/pixel
             "scale_z": float,  # B-scan 深度 mm/pixel
             "scale_y": float}  # B-scan 間距 mm
            ★ 注意命名映射 (你的 → eyepy)：
              你的 scale_x → eyepy scale_x (B-scan lateral)
              你的 scale_z → eyepy scale_y (B-scan depth/axial)
              你的 scale_y → eyepy scale_z (inter-bscan spacing)
        layer_heights: 可選的分層高度，格式：
            {"ILM": [arr_0, arr_1, ...], "BM": [arr_0, arr_1, ...]}
            每個 arr 是長度 X 的 1D float 陣列 (pixel index)。
            ★ 順序必須與 volume_data 對齊（如果 top-first，則也是 top-first）。
            np.nan 表示無效。
        laterality: 'OS' (左眼) 或 'OD' (右眼)
        axial_length_mm: 可選，使用 Littmann-Bennett 校正代替標準 Gullstrand。
        topcon_scan_order_top_first: 若 True，volume_data[0]=superior，
            內部自動 flip 為 eyepy 的 bottom-first 格式。

    Returns:
        標準 eyepy.EyeVolume 物件，可直接使用 .plot(), .save(), [i] 等 API。
    """
    from eyepy.core.eyeenface import EyeEnface
    from eyepy.core.eyemeta import EyeBscanMeta, EyeEnfaceMeta, EyeVolumeMeta
    from eyepy.core.eyevolume import EyeVolume
    from eyepy.io.utils import _compute_localizer_oct_transform

    n_bscans = volume_data.shape[0]

    # ----------------------------------------------------------------
    # 第 0 步：計算度→mm 轉換因子
    # ----------------------------------------------------------------
    if axial_length_mm is not None:
        mm_per_deg = 0.01306 * (axial_length_mm - 1.82)
        print(f"[INFO] Littmann-Bennett 校正: AL={axial_length_mm}mm → 1°≈{mm_per_deg:.4f}mm")
    else:
        mm_per_deg = DEG_TO_MM  # 0.288 mm/°

    fov_mm = SLO_FOV_DEG * mm_per_deg
    half_fov_mm = fov_mm / 2.0
    slo_scale_mm = fov_mm / ir_localizer.shape[1]  # 用實際 SLO 寬度算

    # ----------------------------------------------------------------
    # 第 1 步：轉換 volume_data 為 float32 + 反轉 z 軸
    # ----------------------------------------------------------------
    data = volume_data.astype(np.float32)
    if topcon_scan_order_top_first:
        # Topcon: data[0]=superior → 反轉 → data[0]=inferior (eyepy 約定)
        data = np.flip(data, axis=0).copy()

    # ----------------------------------------------------------------
    # 第 2 步：組裝 EyeBscanMeta（度→mm + 反轉順序）
    # ----------------------------------------------------------------
    # eyepy 內部約定：bscan_meta[0]=最底部, bscan_meta[-1]=最頂部
    # 對應 data[0]=底部, data[-1]=頂部
    if topcon_scan_order_top_first:
        # geometry_coords[0]=頂部 → 需要反轉
        coords_reversed = list(reversed(geometry_coords))
    else:
        coords_reversed = geometry_coords

    bscan_meta_list = []
    for coord in coords_reversed:
        # 度座標 → 以 SLO 左上角為原點的 mm 座標
        start_x_mm = coord['start_x'] * mm_per_deg + half_fov_mm
        start_y_mm = half_fov_mm - coord['start_y'] * mm_per_deg
        end_x_mm = coord['end_x'] * mm_per_deg + half_fov_mm
        end_y_mm = half_fov_mm - coord['end_y'] * mm_per_deg

        bscan_meta_list.append(EyeBscanMeta(
            start_pos=(start_x_mm, start_y_mm),
            end_pos=(end_x_mm, end_y_mm),
            pos_unit='mm',
        ))

    # ----------------------------------------------------------------
    # 第 3 步：讀取/映射 scale 參數
    #   你的命名 → eyepy 命名
    #   scale_x    → scale_x (B-scan 橫向 mm/pixel)
    #   scale_z    → scale_y (B-scan 深度/axial mm/pixel)
    #   scale_y    → scale_z (B-scan 間距 mm)
    # ----------------------------------------------------------------
    eyepy_scale_x = scale_meta['scale_x']  # B-scan 橫向
    eyepy_scale_y = scale_meta['scale_z']  # B-scan 深度 (你的 scale_z → eyepy 的 scale_y)
    eyepy_scale_z = scale_meta['scale_y']  # B-scan 間距 (你的 scale_y → eyepy 的 scale_z)

    volume_meta = EyeVolumeMeta(
        scale_x=eyepy_scale_x,
        scale_y=eyepy_scale_y,
        scale_z=eyepy_scale_z,
        scale_unit='mm',
        laterality=laterality,
        bscan_meta=bscan_meta_list,
        intensity_transform='default',
    )

    # ----------------------------------------------------------------
    # 第 4 步：組裝 EyeEnface (IR localizer)
    # ----------------------------------------------------------------
    enface_meta = EyeEnfaceMeta(
        scale_x=slo_scale_mm,
        scale_y=slo_scale_mm,
        scale_unit='mm',
        modality='NIR',
        laterality=laterality,
        field_size=int(SLO_FOV_DEG),
    )
    localizer = EyeEnface(data=ir_localizer, meta=enface_meta)

    # ----------------------------------------------------------------
    # 第 5 步：計算仿射變換矩陣 (OCT voxel → SLO pixel)
    # ----------------------------------------------------------------
    transformation = _compute_localizer_oct_transform(
        volume_meta, enface_meta, data.shape)

    # ----------------------------------------------------------------
    # 第 6 步：組裝 EyeVolume
    # ----------------------------------------------------------------
    volume = EyeVolume(
        data=data,
        meta=volume_meta,
        localizer=localizer,
        transformation=transformation,
    )

    # ----------------------------------------------------------------
    # 第 7 步：注入分層邊界
    # ----------------------------------------------------------------
    if layer_heights is not None:
        for layer_name, arr_list in layer_heights.items():
            # 從列表組成 2D 陣列
            height_map = np.stack(arr_list, axis=0).astype(np.float32)
            if topcon_scan_order_top_first:
                height_map = np.flip(height_map, axis=0).copy()
            volume.add_layer_annotation(height_map, name=layer_name)
            valid = np.count_nonzero(~np.isnan(height_map))
            total = height_map.size
            print(f"  已注入 {layer_name} 層: 有效率 {valid}/{total} ({valid/total*100:.1f}%)")

    print(f"[OK] 組裝完成! shape={volume.shape}, layers={list(volume.layers.keys())}")
    return volume


# ========================================================================
# 便捷函數：從 CSV + PNG 自動組裝
# ========================================================================

def build_eyevolume(
    series_id: int,
    csv_dir: str,
    image_dir: str,
    volume_data: Optional[np.ndarray] = None,
    ir_localizer: Optional[np.ndarray] = None,
    laterality: str = 'OS',
    axial_length_mm: Optional[float] = None,
    sdb_path: Optional[str] = None,
    image_details_csv: Optional[str] = None,
    bmp_parent_dir: Optional[str] = None,
) -> 'EyeVolume':
    """從 CSV metadata + SLO 圖 + 記憶體中的 volume_data 組裝 EyeVolume。

    這是 build_eyevolume_from_arrays() 的便捷包裝：
    自動從 CSV 讀取 geometry_coords、scale_meta、layer_heights，
    從 PNG 讀取 ir_localizer，你只需提供 volume_data。

    資料來源優先級 (volume_data)：
      1. 直接傳入 volume_data (numpy array)
      2. 從 .sdb 檔案讀取 (提供 sdb_path + image_details_csv)
      3. 從 BMP 資料夾讀取 (提供 bmp_parent_dir)
      4. 自動生成合成資料 (測試用)

    資料來源優先級 (ir_localizer)：
      1. 直接傳入 ir_localizer (numpy array)
      2. 從 .sdb 或 BMP 同步取得
      3. 從 PNG 檔讀取 (image_dir)

    Args:
        series_id: Topcon series ID (例如 36=左眼, 40=右眼)
        csv_dir: CSV 檔案目錄 (如 'csv_output')
        image_dir: 圖檔目錄 (如 'csv_output/ir_images')
        volume_data: shape (N_bscans, H, W) 的 OCT 體積，Topcon 順序 (top-first)。
                     若 None 且有 sdb_path，自動從 .sdb 讀取 uint16。
                     若都沒有，自動生成合成資料。
        ir_localizer: shape (H, W) 的 SLO。若 None，自動載入。
        laterality: 'OS' 或 'OD'
        axial_length_mm: 可選的眼軸長校正
        sdb_path: .sdb 二進位檔路徑。提供後自動讀取 uint16 資料。
        image_details_csv: image_details CSV 路徑 (搭配 sdb_path 使用)
        bmp_parent_dir: 包含各 series 子資料夾的父目錄。
            子資料夾命名為 8 位數字 (如 00000036/)，每個包含 oct_*.bmp + ir_*.bmp。
            會依 series_id 自動找到正確資料夾。

    Returns:
        eyepy.EyeVolume
    """
    if axial_length_mm is not None:
        mm_per_deg = 0.01306 * (axial_length_mm - 1.82)
    else:
        mm_per_deg = DEG_TO_MM

    # ---- 從 CSV 讀取 B-scan 座標 ----
    print(f"[1/5] 讀取 Series {series_id} 的空間座標...")
    pos_df = load_bscan_positions(
        os.path.join(csv_dir, '07_bscans_positions.csv'), series_id)
    n_bscans = len(pos_df)

    # 轉為 geometry_coords 格式
    geometry_coords = []
    for _, row in pos_df.iterrows():
        geometry_coords.append({
            'start_x': row.x1, 'start_y': row.y1,
            'end_x': row.x2, 'end_y': row.y2,
        })

    # ---- 從 CSV 讀取 scaleY (深度 mm/pixel) ----
    print(f"[2/5] 讀取 scale 元資料...")
    meta_df = load_bscan_metadata_csv(
        os.path.join(csv_dir, '06_bscan_metadata.csv'), series_id)
    depth_scale = meta_df['scaleY'].iloc[0]  # mm/pixel (axial)

    # 計算橫向 scale 與間距
    first = pos_df.iloc[0]
    scan_len_deg = np.sqrt((first.x2 - first.x1)**2 + (first.y2 - first.y1)**2)
    bscan_width = meta_df['imgSizeWidth'].iloc[0]
    lateral_scale = (scan_len_deg * mm_per_deg) / bscan_width

    if n_bscans > 1:
        spacing_deg = abs(pos_df.iloc[0].y1 - pos_df.iloc[1].y1)
        bscan_spacing = spacing_deg * mm_per_deg
    else:
        bscan_spacing = 0.0

    scale_meta = {
        'scale_x': lateral_scale,   # B-scan 橫向 mm/pixel
        'scale_z': depth_scale,     # B-scan 深度 mm/pixel (你的命名: scale_z)
        'scale_y': bscan_spacing,   # B-scan 間距 mm (你的命名: scale_y)
    }
    print(f"  scale_x={lateral_scale*1000:.2f} μm/px, "
          f"scale_z(depth)={depth_scale*1000:.2f} μm/px, "
          f"scale_y(spacing)={bscan_spacing*1000:.2f} μm")

    # ---- 處理影像資料 (SLO + volume_data) ----
    # 資料來源優先級:
    #   volume_data: 傳入參數 > .sdb > BMP 資料夾 > 合成
    #   ir_localizer: 傳入參數 > .sdb/.bmp 同步 > PNG
    ir_loc = ir_localizer  # 可能為 None，稍後處理
    bscan_height = meta_df['imgSizeX'].iloc[0]  # Topcon imgSizeX = depth samples

    if volume_data is None and sdb_path is not None and image_details_csv is not None:
        print(f"[3/5] 從 .sdb 讀取 uint16 volume + SLO...")
        sdb_result = read_sdb_images(sdb_path, image_details_csv, series_id)
        volume_data = sdb_result['volume_data']
        if ir_loc is None:
            ir_loc = sdb_result['slo_data']
        print(f"  Volume: {volume_data.shape} {volume_data.dtype} "
              f"(range=[{volume_data.min()}, {volume_data.max()}])")
    elif volume_data is None and bmp_parent_dir is not None:
        print(f"[3/5] 從 BMP 資料夾讀取 (series {series_id})...")
        bmp_result = read_bmp_by_series(bmp_parent_dir, series_id, csv_dir)
        volume_data = bmp_result['volume_data']
        if ir_loc is None:
            ir_loc = bmp_result['slo_data']
    elif volume_data is None:
        print(f"[3/5] [WARNING] 未提供 volume_data，生成合成測試資料 "
              f"({n_bscans}×{bscan_height}×{bscan_width})...")
        volume_data = np.zeros((n_bscans, bscan_height, bscan_width), dtype=np.uint8)
        for i in range(n_bscans):
            gradient = np.linspace(30, 220, bscan_height).reshape(-1, 1)
            noise = np.random.randint(0, 15, (bscan_height, bscan_width))
            volume_data[i] = np.clip(gradient + noise, 0, 255).astype(np.uint8)
    else:
        print(f"[3/5] 使用提供的 volume_data, shape={volume_data.shape}")
        if volume_data.shape[0] != n_bscans:
            raise ValueError(
                f"volume_data 的 B-scan 數 ({volume_data.shape[0]}) "
                f"與 CSV 中的數量 ({n_bscans}) 不一致")

    # SLO fallback: 從 PNG 讀取
    if ir_loc is None:
        print(f"[4/5] 從 PNG 讀取 SLO 底圖...")
        ir_loc = load_slo_image(image_dir, series_id)
    else:
        print(f"[4/5] SLO 已準備, shape={ir_loc.shape} {ir_loc.dtype}")

    # ---- 從 CSV 讀取分層邊界 ----
    print(f"[5/5] 讀取分層邊界...")
    layer_boundaries_csv = os.path.join(csv_dir, '12_layer_boundaries.csv')
    if os.path.exists(layer_boundaries_csv):
        raw_layers = load_layer_boundaries(layer_boundaries_csv, series_id)
        # 轉為 build_eyevolume_from_arrays 需要的 list 格式，
        # 但此時已經是反轉過的 (bottom-first)，需要 re-flip 因為
        # build_eyevolume_from_arrays 會再 flip 一次
        layer_heights = {}
        for name, arr in raw_layers.items():
            # raw_layers 已經 flip 過了，恢復為 top-first 讓核心函數再 flip
            arr_topfirst = np.flip(arr, axis=0).copy()
            layer_heights[name] = [arr_topfirst[i] for i in range(arr_topfirst.shape[0])]
    else:
        layer_heights = None

    # ---- 呼叫核心組裝函數 ----
    return build_eyevolume_from_arrays(
        volume_data=volume_data,
        ir_localizer=ir_loc,
        geometry_coords=geometry_coords,
        scale_meta=scale_meta,
        layer_heights=layer_heights,
        laterality=laterality,
        axial_length_mm=axial_length_mm,
        topcon_scan_order_top_first=True,
    )


# ========================================================================
# 雙眼批次建構
# ========================================================================

def build_both_eyes(
    csv_dir: str = 'csv_output',
    image_dir: str = 'csv_output/ir_images',
    volume_data_os: Optional[np.ndarray] = None,
    volume_data_od: Optional[np.ndarray] = None,
    sdb_dir: Optional[str] = None,
    image_details_csv: Optional[str] = None,
    bmp_parent_dir: Optional[str] = None,
) -> dict[str, 'EyeVolume']:
    """一次組裝左右眼的 EyeVolume 物件。

    Args:
        csv_dir: CSV 目錄
        image_dir: 圖檔目錄
        volume_data_os: 左眼的 OCT 體積 (可選)
        volume_data_od: 右眼的 OCT 體積 (可選)
        sdb_dir: 包含 .sdb 檔案的目錄 (可選，提供後自動讀取 uint16)
        image_details_csv: image_details CSV 路徑 (搭配 sdb_dir 使用)

    Returns:
        {'OS': EyeVolume, 'OD': EyeVolume}
    """
    results = {}

    # 自動定位 .sdb 檔案
    sdb_os = None
    sdb_od = None
    if sdb_dir is not None:
        sdb_os_path = Path(sdb_dir) / '00000036.sdb'
        sdb_od_path = Path(sdb_dir) / '00000040.sdb'
        if sdb_os_path.exists():
            sdb_os = str(sdb_os_path)
        if sdb_od_path.exists():
            sdb_od = str(sdb_od_path)

    print("=" * 60)
    print("建構左眼 (OS) — Series 36")
    print("=" * 60)
    results['OS'] = build_eyevolume(
        series_id=36, csv_dir=csv_dir, image_dir=image_dir,
        volume_data=volume_data_os, laterality='OS',
        sdb_path=sdb_os, image_details_csv=image_details_csv,
        bmp_parent_dir=bmp_parent_dir)

    print()

    print("=" * 60)
    print("建構右眼 (OD) — Series 40")
    print("=" * 60)
    results['OD'] = build_eyevolume(
        series_id=40, csv_dir=csv_dir, image_dir=image_dir,
        volume_data=volume_data_od, laterality='OD',
        sdb_path=sdb_od, image_details_csv=image_details_csv,
        bmp_parent_dir=bmp_parent_dir)

    return results


# ========================================================================
# 驗證與視覺化
# ========================================================================

def verify_volume(volume: 'EyeVolume', title: str = 'EyeVolume') -> None:
    """視覺化驗證 EyeVolume 的空間對齊。

    產生 4 張子圖：
    1. SLO + B-scan 位置線
    2. 中間 B-scan 的截面 + 層分割
    3. ILM/BM 厚度熱力圖 (enface 投影)
    4. 3D 層表面圖 (如有 matplotlib 3D 支援)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{title} — Spatial Alignment Verification', fontsize=14)

    # --- (1) SLO + B-scan 位置線 ---
    ax1 = axes[0, 0]
    volume.plot(
        ax=ax1,
        bscan_positions=True,
        scalebar='botleft',
        watermark=False,
    )
    ax1.set_title('SLO + B-scan Positions')

    # --- (2) 中間 B-scan + 層分割線 ---
    ax2 = axes[0, 1]
    mid_idx = len(volume) // 2
    bscan = volume[mid_idx]
    bscan.plot(ax=ax2, layers=True, watermark=False)
    ax2.set_title(f'B-scan #{mid_idx} (Central Slice) + Layers')

    # --- (3) 視網膜厚度熱力圖 ---
    ax3 = axes[1, 0]
    if 'ILM' in volume.layers and 'BM' in volume.layers:
        ilm = volume.layers['ILM'].data  # (n_bscans, width)
        bm = volume.layers['BM'].data
        # 厚度 = BM − ILM (以 pixel 為單位，乘以 scale_y 轉 mm)
        thickness_px = bm - ilm
        thickness_um = thickness_px * volume.scale_y * 1000  # μm
        # 遮罩無效區域
        thickness_um = np.where(np.isnan(thickness_px), np.nan, thickness_um)
        im = ax3.imshow(thickness_um, cmap='hot', aspect='auto',
                        interpolation='nearest')
        plt.colorbar(im, ax=ax3, label='Thickness (um)')
        ax3.set_xlabel('A-scan Position')
        ax3.set_ylabel('B-scan Index (0=inferior)')
        ax3.set_title('Total Retinal Thickness (BM - ILM)')
    else:
        ax3.text(0.5, 0.5, 'No ILM/BM Layer Data',
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Retinal Thickness (N/A)')

    # --- (4) 3D 層表面圖 ---
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    if 'ILM' in volume.layers and 'BM' in volume.layers:
        ilm = volume.layers['ILM'].data
        bm = volume.layers['BM'].data
        n_bscans, width = ilm.shape

        # 建構 3D 網格座標
        x = np.arange(width) * volume.scale_x    # mm (A-scan 橫向)
        z = np.arange(n_bscans) * volume.scale_z  # mm (B-scan 間距)
        X, Z = np.meshgrid(x, z)

        # Y 軸 = 層的深度位置 (pixel → mm)
        ilm_mm = ilm * volume.scale_y
        bm_mm = bm * volume.scale_y

        # 每隔幾個像素採樣以加速繪製
        step_x = max(1, width // 64)
        step_z = max(1, n_bscans // 25)

        ax4.plot_surface(
            X[::step_z, ::step_x],
            Z[::step_z, ::step_x],
            ilm_mm[::step_z, ::step_x],
            alpha=0.5, color='blue', label='ILM')
        ax4.plot_surface(
            X[::step_z, ::step_x],
            Z[::step_z, ::step_x],
            bm_mm[::step_z, ::step_x],
            alpha=0.5, color='red', label='BM')

        ax4.set_xlabel('A-scan (mm)')
        ax4.set_ylabel('B-scan Spacing (mm)')
        ax4.set_zlabel('Depth (mm)')
        ax4.set_title('ILM (Blue) / BM (Red) 3D Surface')
        ax4.invert_zaxis()  # 深度翻轉：淺處在上
    else:
        ax4.text2D(0.5, 0.5, 'No ILM/BM Layer Data',
                   ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}_verification.png', dpi=150, bbox_inches='tight')
    print(f"Verification saved: {title.replace(' ', '_')}_verification.png")
    plt.show()


def verify_transform_alignment(volume: 'EyeVolume', title: str = '') -> None:
    """額外驗證：在 SLO 上標註四角對齊點，確認仿射變換正確。"""
    import matplotlib.pyplot as plt
    from skimage import transform as sktf

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(volume.localizer.data, cmap='gray')

    t = volume.localizer_transform
    n_bscans = len(volume)
    width = volume.shape[2]

    # OCT 四角 → 透過變換映射到 SLO 上
    corners_oct = np.array([
        [0, 0],            # Top left (col=0, row=0)
        [width - 1, 0],   # Top right
        [0, n_bscans - 1], # Bottom left
        [width - 1, n_bscans - 1],  # Bottom right
    ], dtype=float)

    corners_slo = t(corners_oct)
    ax.plot(corners_slo[:, 0], corners_slo[:, 1], 'r+', markersize=15, mew=2)

    # 標註每張 B-scan 的起終點
    for i in range(n_bscans):
        start = t(np.array([[0, i]]))[0]
        end = t(np.array([[width - 1, i]]))[0]
        ax.plot([start[0], end[0]], [start[1], end[1]],
                'g-', linewidth=0.5, alpha=0.6)

    ax.set_title(f'{title} — Affine Transform Alignment Check')
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}_transform_check.png', dpi=150)
    print(f"Transform check saved: {title.replace(' ', '_')}_transform_check.png")
    plt.show()


# ========================================================================
# 主程式入口
# ========================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

    # --- 自動偵測 BMP 來源 ---
    # 優先級: CGMH_EXTRACTED (per-series folders) > test/ (flat folder)
    bmp_parent = None
    vol_os_data = None
    vol_od_data = None

    # 嘗試 CGMH_EXTRACTED 目錄 (每個 series 一個子資料夾)
    cgmh_dir = os.path.join(os.path.dirname(__file__), 'CGMH_EXTRACTED')
    if os.path.isdir(cgmh_dir):
        # 找到第一個 .pat 資料夾
        pat_dirs = sorted([d for d in os.listdir(cgmh_dir)
                          if os.path.isdir(os.path.join(cgmh_dir, d))])
        if pat_dirs:
            bmp_parent = os.path.join(cgmh_dir, pat_dirs[0])
            print(f"[main] 偵測到 CGMH_EXTRACTED/{pat_dirs[0]}，使用 per-series BMP 讀取")

    # 退回 test/ 目錄 (flat folder)
    if bmp_parent is None:
        test_dir = os.path.join(os.path.dirname(__file__), 'test')
        if os.path.exists(os.path.join(test_dir, 'oct_0000_512x496.bmp')):
            print("[main] 偵測到 test/ 目錄中的 BMP 檔案，讀取中...")
            bmp_result = read_bmp_volume(test_dir, n_bscans=25)
            vol_os_data = bmp_result['volume_data']

    # --- 建構雙眼 ---
    volumes = build_both_eyes(
        csv_dir='csv_output',
        image_dir='csv_output/ir_images',
        volume_data_os=vol_os_data,
        volume_data_od=vol_od_data,
        bmp_parent_dir=bmp_parent,
    )

    # --- 驗證 ---
    for eye, vol in volumes.items():
        print(f"\n{'='*40}")
        print(f"驗證 {eye} 眼")
        print(f"{'='*40}")
        print(f"  Volume shape: {vol.shape}")
        print(f"  Scale (z, y, x): {vol.scale}")
        print(f"  Layers: {list(vol.layers.keys())}")
        print(f"  Localizer shape: {vol.localizer.data.shape}")
        print(f"  Transform matrix:\n{vol.localizer_transform.params}")
        verify_volume(vol, title=f'{eye}')
        verify_transform_alignment(vol, title=f'{eye}')

    # --- 可選：儲存為 eyepy 原生格式 ---
    for eye, vol in volumes.items():
        save_path = f'topcon_{eye}_volume.eyepy'
        vol.save(save_path)
        print(f"\n已儲存: {save_path}")

    # ================================================================
    # ★ 使用範例：插入你自己的 OCT 體積資料 ★
    # ================================================================
    # 假設你從 Topcon 二進位檔萃取出以下 numpy 陣列：
    #
    # import numpy as np
    # my_volume = np.load('my_topcon_oct.npy')  # shape (25, 496, 512), uint8
    # my_ir = np.array(Image.open('my_slo.png').convert('L'))  # shape (768, 768)
    #
    # 方法 A：直接用 build_eyevolume 便捷函數
    # vol = build_eyevolume(
    #     series_id=36,
    #     csv_dir='csv_output',
    #     image_dir='csv_output/ir_images',
    #     volume_data=my_volume,  # ← 插入你的實際資料
    #     laterality='OS',
    # )
    #
    # 方法 B：完全手動控制 (適合非標準座標系)
    # vol = build_eyevolume_from_arrays(
    #     volume_data=my_volume,
    #     ir_localizer=my_ir,
    #     geometry_coords=[
    #         {"start_x": -10, "start_y": 10, "end_x": 10, "end_y": 10},
    #         {"start_x": -10, "start_y": 9.167, "end_x": 10, "end_y": 9.167},
    #         # ... 共 25 筆
    #     ],
    #     scale_meta={
    #         "scale_x": 0.01125,  # B-scan 橫向 mm/px
    #         "scale_z": 0.003872, # B-scan 深度 mm/px (你的命名)
    #         "scale_y": 0.24,     # B-scan 間距 mm (你的命名)
    #     },
    #     layer_heights={
    #         "ILM": [ilm_scan0, ilm_scan1, ...],  # 每個是長度 512 的 1D array
    #         "BM":  [bm_scan0,  bm_scan1,  ...],
    #     },
    #     laterality='OS',
    # )
