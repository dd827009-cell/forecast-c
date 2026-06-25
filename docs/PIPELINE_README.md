# heyex_pipeline — Heidelberg `.pat` → 訓練用 HDF5

將一批 Heidelberg HEYEX `.pat` 資料夾（內含 `.sdb`/`.edb`/`.pdb` 二進位檔）
轉換為每個 eye-visit 一份的 HDF5，作為 3D OCT Foundation Model 訓練 pipeline
的輸入。

範圍：

- **包含** — 二進位解析、per-series macular cube 篩選、layer 抽取、fovea
  估計、QC flags、確定性輸出路徑、平行 batch runner、parquet manifest、JSONL
  failure log。
- **不包含** — 模型訓練、DataLoader、SSL heads。

## 安裝

三個硬性依賴 + 視覺化工具依賴：

```
numpy
h5py
pandas
pyarrow
matplotlib     # 僅 tools/verify_sample.py 用到
```

```bash
python -m pip install numpy h5py pandas pyarrow matplotlib
```

Python ≥ 3.10。Windows 與 Linux 皆支援。

## 批次執行

```
python -m heyex_pipeline \
    --input /path/to/sdb_root \
    --output /path/to/h5_output \
    --workers 16 \
    --manifest-checkpoint-interval 10000
```

參數說明：

| Flag | 說明 |
|---|---|
| `--input` | 會遞迴搜尋底下所有 `*.pat` 目錄。 |
| `--output` | 輸出根目錄。內含 `{h[:2]}/{h[2:4]}/{patient_id}/...h5`、`manifest.parquet`、`failures.jsonl`。 |
| `--workers` | ProcessPoolExecutor 大小；1 時在主程序內執行。 |
| `--manifest-checkpoint-interval N` | 每處理 N 筆 flush 一次 parquet。 |
| `--dry-run` | 只列出會寫入哪些檔案，不動硬碟（manifest 也不會更新）。 |
| `--verify-samples N` | 跑完後隨機選 N 個 `.h5` 產出 `verify_sample` PNG 到 `{output}/_verify/`。 |
| `-v`/`--verbose` | INFO-level log。 |

### 輸出目錄結構

```
{out_root}/
  <h1>/<h2>/<patient_id>/<visit_id>_<laterality>.h5
  manifest.parquet
  failures.jsonl
```

其中 `<h1><h2> = sha1(patient_id.encode()).hexdigest()[:4]`。

`<visit_id>` 為 `YYYYMMDDTHHMMSS`（UTC，取第一張 B-scan 的 acquisition
time），`<laterality>` 為 `OD`（右眼）或 `OS`（左眼），從 `EyeData.eyeSide`
解出。

### Idempotency（重跑跳過）

每次執行都會比對既有 HDF5 的 `parser_version` **major 版號**：相符就跳過。
Backward-compatible 的修正升 minor；不相容改動必須升 major 強制重建。唯一
的正式版號在 `heyex_pipeline/version.py`。

## 單樣本視覺驗證

```
python tools/verify_sample.py \
    /path/to/out/<h1>/<h2>/<patient>/<visit>_OD.h5 \
    -o /tmp/verify.png
```

四格驗證圖：

1. IR localizer 疊所有 B-scan 的 A-scan 位置線 + fovea 十字。
2. 中央 B-scan 疊 ILM / RPE_BM / BM_true。
3. Thickness（RPE_BM − ILM）heatmap。
4. ILM 與 RPE_BM 的 3D surface。

## segType 7 稽核

```
python tools/diagnose_layer7.py --input /data/raw --output seg7.csv
```

對每個 series 統計 seg_type==7（真 BM）chunks 數量，以及 finite A-scan
數 / FLT_MAX 哨兵數。這份資料集中 seg_type 7 幾乎全部是哨兵；若未來換了
batch 開始出現有效值，用這支工具幾分鐘內就可確認。

## Troubleshooting

### `failures.jsonl` 出現 `no_entries_found`

`.pat` 內沒有任何符合規格的 `.sdb/.edb/.pdb`（byte 0 的 `CMDb` magic
或 `0x24` 的 `MDbMDir`）。通常代表檔案截斷；HEYEX 也讀不進去。

### 全部都 `not_macular_volume:*`

多半是不支援的掃描模式（只有 line scans、20 度 cube、star、OCTA）。
`heyex_pipeline/filters.py` 列出所有判定條件：scan pattern、尺寸白名單、
center offset、OCTA 排除。

### `missing_layer_segmentation`

HEYEX 這個 series 沒有產 ILM（type 5）與 RPE_BM（type 2）。我們不用
fallback：這兩層是臨床人工驗證過的 anatomical prior，是模型唯一可信的
監督訊號，少一層就直接丟掉。

### Manifest 出現 `fovea_valid=False`

中央 5 張 B-scan 裡找不到 3×3 鄰域且有效率 ≥ 0.9 的候選點。樣本仍會
寫入，但加 `fovea_undetectable` soft flag，`fovea_ir_x/y = NaN`。

### `parser_version` major 不相符導致全部被跳過

這就是 idempotency 的正常行為——上次執行留下的是不同 major 的檔。
要嘛清掉 output、要嘛退回對應版本的 parser。

### Worker 崩潰

用 `--workers 1` 可以讓 `.pat` 在主程序中執行，比較容易拿到完整
traceback。

## 測試

```
pytest                               # 全部 114 個 test，~3 秒
pytest tests/test_integration.py     # 合成 .pat 走完整 pipeline
pytest tests/test_schema_contract.py # 強制檢查 E.3 欄位 / dtype
```

[tests/_synthetic_pat.py](tests/_synthetic_pat.py) 會在 `tmp_path` 內組出
一個極簡但合法的 `.sdb`，讓 integration 與 CLI 測試能用真正的 binary
parser 跑完，而完全不需要任何病人資料。
