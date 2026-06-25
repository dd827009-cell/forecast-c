# stage0 — OCT 資料準備 pipeline（M1–M7）

把轉好的 `.h5`（來自 `../pdb_to_h5/`）處理成 OCTCube 持續預訓練用的 **WebDataset shard** +
`manifest.parquet` + `norm_stats.json`。自足、純 CPU、不依賴 torch。

## 模組

| 模組 | 角色 | 產出 / 函式 |
|---|---|---|
| `_version.py` | 前處理版本字串（單一真相） | `STAGE0_VERSION` |
| `m1_build_index.py` | 掃 `.h5`、讀 attrs/shape、scale 全域常數驗證(A1) | `index.parquet` |
| `m2_transform.py` | 強度轉換 sqrt+P15地板+pctl[1,99.9]→[0,1]（函式庫 `transform_volume`） | — |
| `m3_geometry.py` | OS翻轉 + FOV置中裁512 + resize256（變體A）；層 native；mask 256/512（函式庫 `process_volume`） | — |
| `m4_qc.py` | QC（image_quality/valid_ratio/n_bscans/w_consistent） | `index_qc.parquet` |
| `m5_meanstd.py` | 全資料 mean/std（train-only 防洩漏） | `norm_stats.json` |
| `m6_split.py` | patient-level split 96/2/2（+A2 縱向覆蓋；`--prev-manifest` 穩定增量） | `manifest.parquet` |
| `m7_pack.py` | 打包 WebDataset tar（volume+valid_mask+meta；版本感知增量/sha256/壓縮可選） | shard + `manifest_packed.parquet` |
| `m7b_pack.py` | **M7b 抽存評估資產**(ir/ilm/rpe/valid/ascan_pos_ir per-eye `.npz`,供 Stage2/世界模型) | assets + `manifest_assets.parquet` |
| `verify_m5_m6.py` | M5/M6 獨立交叉驗證（23 項） | — |
| `verify_m7.py` | M7 打包→讀回→比對→清理（14 項） | — |
| `verify_m7b.py` | **M7b 抽存讀回驗證**(shape/dtype/NaN/**ILM≤RPE 層不交叉**) | — |

## 相依
```bash
pip install numpy opencv-python h5py pandas pyarrow tqdm
# matplotlib 選用 (僅 m2/m3 的 _verify 視覺化)
```

## 執行順序（⚠️ M6 要在 M5 之前——M5 只在 train split 上算）
```bash
H5_DIR=<.h5 目錄>   # 來自 ../pdb_to_h5
python stage0/m1_build_index.py --root "$H5_DIR" --out stage0/index.parquet --workers 32
python stage0/m4_qc.py        --index stage0/index.parquet --out stage0/index_qc.parquet
python stage0/m6_split.py     --index stage0/index_qc.parquet --out stage0/manifest.parquet
python stage0/m5_meanstd.py   --manifest stage0/manifest.parquet --splits train --workers 32
python stage0/verify_m5_m6.py
python stage0/m7_pack.py       --manifest stage0/manifest.parquet --raw-root "$H5_DIR" \
                               --out-dir <SHARD_DIR> --vols-per-shard 512 --workers 32
python stage0/verify_m7.py     --manifest stage0/manifest.parquet
```

## 增量（資料陸續匯出）
- M6：`--prev-manifest <上一版>` → 既有病人凍結、只切新病人（含增量自檢）。
- M7：重跑即可，讀 `manifest_packed` 跳過已打包（且同 `transform_version`）；改配方(版本變)會重打包並警告。
- M1/M4：重掃即冪等。

## 已內建的關鍵決策
- 裁切＝**變體A**（FOV 廣域降採，重建實證優於黃斑置中、貼 OCTCube 預訓練分布）。
- 存 **25-slice fp16 [0,1]**；**正規化與 25→60 內插留給 dataloader**（改參數不必重打包）。
- 層採方案C（native 不 resize）；split 96/2/2（大資料慣例）。

詳見 `../RUNBOOK_NAS.md`（全量流程）與 `../OCTCube-M_持續預訓練_Pipeline.md`（總規劃）。
