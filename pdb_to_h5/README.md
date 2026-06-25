# pdb_to_h5 — Heidelberg `.pat` → `.h5` 轉換器（parser v3.0.0）

把 Heidelberg Spectralis 原始 `.pat` 目錄（內含 `.pdb`/`.sdb`/`.edb`）批次轉成訓練用的
`.h5`，供後續 Stage 0（`stage0/` 的 M1–M7）使用。

> 這是 `test1/heyex_pipeline` 的**獨立複本**，抽出來方便日後直接使用（同源、同 parser v3.0.0）。

## 結構
- `heyex_pipeline/` — 轉換器套件（cli / sample_extractor / hdf5_writer / layers / fovea / qc …）
- `existing/` — 底層 Heidelberg 解析器（`sample_extractor.py` 會把此資料夾加進 `sys.path` 並
  `from export_e2e_csv import ...`，**必須與 `heyex_pipeline/` 同層保留**）
- `tools/` — `verify_sample.py`（僅 `--verify-samples` 會 lazy import）

## 相依
Python 3.10+，安裝：
```bash
pip install numpy h5py pandas simplejson iopath
```

## 用法
```bash
cd pdb_to_h5

# 先 dry-run 確認掃得到 .pat、不寫檔
python -m heyex_pipeline --input "<.pat 根目錄>" --output "<.h5 輸出目錄>" --dry-run -v

# 正式批次轉換（平行；可續跑）
python -m heyex_pipeline --input "<.pat 根目錄>" --output "<.h5 輸出目錄>" \
    --workers 32 --manifest-checkpoint-interval 10000 --verify-samples 20
```

參數：`--input`（遞迴找 .pat）/ `--output`（.h5 + manifest + failures.jsonl）/ `--workers` /
`--manifest-checkpoint-interval`（中斷可續跑）/ `--dry-run` / `--verify-samples N` / `-v`。

## 行為
- **冪等/可續跑**：已轉過、且 `parser_version` major 相符的 `.h5` 會被跳過；中斷後重跑只補未完成的。
- 產出每個 (visit, eye) 一個 `.h5`，schema：
  `volume / ir / ilm_y / rpe_bm_y / valid_ascan_mask / ascan_pos_ir / image_quality_per_bscan`
  + attrs（`parser_version=3.0.0` / `patient_id` / `laterality` / `visit_id` / `longitudinal_key` / `scale_*` / `fovea_ir_x,y` …）。
- 之後接 `stage0/m1_build_index.py --root <.h5 輸出目錄>` → M4 → M6 → M5 → M7。詳見 `../RUNBOOK_NAS.md`。

## 驗證
獨立 dry-run 已實測（掃到 21 個 .pat、可轉 81 顆、0 失敗）：
```bash
python -m heyex_pipeline --input "../CGMH RAW DATA" --output /tmp/_dryrun --dry-run -v
```
