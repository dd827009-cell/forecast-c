# HDF5 Schema（parser_version 1.x）

每個處理成功的 eye-visit sample 對應一個 HDF5 檔。

## 路徑

```
{out_root}/{h[:2]}/{h[2:4]}/{patient_id}/{visit_id}_{laterality}.h5
其中 h = sha1(patient_id.encode("utf-8")).hexdigest()
```

`visit_id` 為 `YYYYMMDDTHHMMSS`，UTC，取自第一張 B-scan 的 acquisition
time，檔名安全（沒有冒號）。`laterality` 為 `OD` 或 `OS`。

## Datasets

| Dataset | Shape | dtype | 備註 |
|---|---|---|---|
| `/volume` | `(D, H, W)` | uint16 | `chunks=(1, H, W)`，`compression="lzf"` |
| `/ir` | `(H_ir, W_ir)` | uint8 | 有 IR 時用 `compression="lzf"`；沒有 IR 時 shape `(0, 0)` |
| `/ilm_y` | `(D, W)` | float32 | 無效 A-scan 為 NaN（segType 5） |
| `/rpe_bm_y` | `(D, W)` | float32 | 無效 A-scan 為 NaN（segType 2） |
| `/bm_true_y` | `(D, W)` | float32 | segType 7 完全沒有有效值時**不存在** |
| `/valid_ascan_mask` | `(D, W)` | bool | `ilm_y` 與 `rpe_bm_y` 都 finite 才為 True |
| `/ascan_pos_ir` | `(D, W, 2)` | float32 | `[..., 0]=x_px`、`[..., 1]=y_px`；無 IR 時 shape `(0, 0, 0)` |
| `/image_quality_per_bscan` | `(D,)` | float32 | HEYEX 每張 B-scan 的 Q 值，optional |
| `/line_scans` | group | — | 一律存在；沒有 line scan 時底下 0 個 dataset |
| `/line_scans/scan_NNN` | `(H, W)` | uint8 | 含 attrs：`pattern`、`ir_x1/y1/x2/y2`、`image_quality` |

`D` = B-scan 數，`H` = axial depth（Spectralis macular cube 一律 496），
`W` = 每張 B-scan 的 A-scan 數（384 / 512 / 768）。

`ilm_y` / `rpe_bm_y` / `bm_true_y` 的 y 值是 B-scan 的 row 座標
（0 = 最上），與 HEYEX 原生慣例一致。

## Root attributes（全部必備）

| 名稱 | Type | 說明 |
|---|---|---|
| `patient_id` | str | 醫院病人 ID（來自 `PatientData.id`）。 |
| `visit_date` | str | ISO-8601 UTC（例如 `2021-05-06T11:22:33+00:00`）。 |
| `visit_id` | str | 檔名安全的 `YYYYMMDDTHHMMSS`。 |
| `acquisition_time_utc` | str | 與 `visit_date` 相同。 |
| `laterality` | str | `OD` 或 `OS`。 |
| `n_bscans` | int32 | `D`。 |
| `bscan_height` | int32 | `H`。 |
| `bscan_width` | int32 | `W`。 |
| `scale_axial_um_per_px` | float32 | `BScanMeta.scaleY × 1000`（典型值 ≈ 3.87）。 |
| `scale_lateral_mm_per_px` | float32 | B-scan 長度 / `W`。 |
| `scale_bscan_spacing_mm` | float32 | 相鄰 B-scan 的 \|ΔposY1\| 中位數 × 0.288 mm/°。 |
| `bscan_spacing_deg_per_index` | float32 | 同上，但以度為單位。 |
| `image_quality` | float32 | per-B-scan Q 值的中位數。 |
| `valid_ascan_count` | int32 | `valid_ascan_mask` 中 True 的總數。 |
| `valid_ascan_ratio` | float32 | `valid_ascan_count / (D*W)`。 |
| `longitudinal_key` | str | `"{patient_id}::{laterality}"`（可依此聚合同一眼的多次就診）。 |
| `visit_uid` | str | `"{patient_id}::{visit_id}"`（每次就診唯一）。 |
| `segmentation_types_available` | str | 固定順序：`"ILM,RPE_BM"` 或 `"ILM,RPE_BM,BM_true"`。 |
| `has_ir` | bool | `ir.size > 0`。 |
| `has_line_scans` | bool | `n_line_scans > 0`。 |
| `n_line_scans` | int32 | `/line_scans/scan_*` dataset 數。 |
| `fovea_ir_x` | float32 | Fovea 的 IR 像素 x 座標；偵測不到為 NaN。 |
| `fovea_ir_y` | float32 | Fovea 的 IR 像素 y 座標；偵測不到為 NaN。 |
| `parser_version` | str | `"1.0.0"`。升 major 要重建，升 minor 相容。 |
| `source_sdb_path` | str | 原始 `.sdb` 的絕對路徑。 |
| `source_edb_path` | str | `.edb` 路徑；不存在為 `""`。 |
| `source_pdb_path` | str | `.pdb` 路徑；不存在為 `""`。 |
| `flags` | str | 逗號分隔的 soft QC flags；通過所有檢查為 `""`。 |

## Flags

### Soft（仍然寫入 HDF5）

| Flag | 觸發條件 |
|---|---|
| `low_layer_coverage` | `valid_ascan_ratio < 0.70` |
| `low_quality` | `image_quality < 15` |
| `irregular_bscan_spacing` | \|ΔposY1\| 間距的 CV > 0.10 |
| `extreme_thickness` | 中位厚度不在 [50, 150] px |
| `missing_ir` | 沒有 IR localizer |
| `fovea_undetectable` | 中央 B-scan 找不到有效 3×3 鄰域 |

### Hard failures（**不**寫 HDF5；寫一行到 `failures.jsonl`）

| Reason | Stage | 意義 |
|---|---|---|
| `no_entries_found` | `open_sdb` | `.pat` 內沒有可解析的 entry。 |
| `no_bscan_meta` | `read_series` | 這個 series 沒有 `BScanMetaData`。 |
| `no_oct_images` | `read_series` | 這個 series 沒有 `subID==1` 的 OCT 影像。 |
| `bad_image_header` / `corrupt_pixel_data:*` | `read_series` | 影像 payload 不完整或尺寸不一致。 |
| `not_macular_volume:*` | — | 靜默過濾，不記 `failures.jsonl`。 |
| `undetermined_laterality` | `validate` | `EyeData.eyeSide` 缺失或無法辨識。 |
| `missing_layer_segmentation` | `validate` | ILM 或 RPE_BM 完全沒有有效值。 |
| `qc_fail_rpe_below_ilm` | `validate` | 太多 A-scan 出現 RPE_BM ≥ ILM（上下顛倒）。 |
| `inverted_h_axis` | `validate` | 啟發式判定 H 軸被翻轉。 |
| `write_exception` | `write` | `h5py` / `os.replace` 失敗（磁碟滿、權限不足）。 |

## Line-scan datasets

cube 一起採集的矢狀 / 放射線 scan。每一個是 2-D uint8 影像，attributes
與 HEYEX 原生 `BScanMeta` 對齊：

- `pattern` — `ScanPattern` 字串（例如 `"Line"`、`"Radial"`）。
- `ir_x1`, `ir_y1`, `ir_x2`, `ir_y2` — 線段端點的 IR 像素座標。
- `image_quality` — 該條 line scan 的 Q 值。

## Manifest（`manifest.parquet`）

每個成功寫入的 HDF5 對應一 row。欄位順序以
`heyex_pipeline.manifest.MANIFEST_COLUMNS` 為準。

## Failure log（`failures.jsonl`）

每個 hard failure 一個 JSON object。keys：

```
source_sdb_path, source_edb_path, study_id, series_id, patient_id,
failure_stage, reason, exception, parser_version, timestamp_utc
```

`failure_stage ∈ { open_sdb, read_series, parse_layers, validate, write }`。

## 相容性

- Reader 端**必須**檢查 `parser_version.split(".")[0]`，major 不符就拒絕
  該檔。
- Minor 版升級可新增 dataset；同一個 major 內既有 dataset 的 shape / dtype
  不會變。
- Optional datasets（`/bm_true_y`、`/image_quality_per_bscan`）可能不存在，
  reader **必須**用 `if "name" in h5:` 判斷。
