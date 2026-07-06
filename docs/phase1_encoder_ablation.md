# Phase 1 Encoder 存亡實驗 + 證成清單

> 上機做實驗時照這張表一格一格填。baseline 實作骨架在 [`forecast_c/model/baselines.py`](../forecast_c/model/baselines.py)。

## 定位
- **真正的 encoder** = 多教師蒸餾 student(DINOv3 + RETFound + MedSAM2 + MIM) → 2D student → **3D adapter(Conv3d depth-only 3×1×1 + residual) + 空間 JEPA**。
- **OCTCube = baseline / 凍結前閘門**（不是模型本體，是最關鍵對照）。
- 路線合理（design C 原創），但要扛的是**「證成」**——每個元件都要用消融證明有用，不能假設。

## 四個必須扛的「證成」
1. **student 必打贏 OCTCube** — 別把 OCTCube 當鷹架丟掉；它是你最關鍵的 baseline。沒打贏 → 不如直接用 OCTCube。
2. **逐教師消融 = Phase 1 主結果** — 證「多教師 > 單教師」（RETFound-only vs +DINOv3 vs +MedSAM2 vs 全部）。
3. **3D adapter depth-only 3×1×1 太薄** — 25 稀疏 B-scan、層間距大，薄 conv 跨層上下文可能不足 → 備「厚版」(多層 depth conv / 輕量 cross-slice attention) 消融。
4. **空間 JEPA 要贏 MIM-only** — 否則原創點沒立起來。

## 教師對照（楊瀚博 vs 我們）
| 功能軸 | 楊瀚博 | 我們 | 一致? | 備註 |
|---|---|---|---|---|
| 通用視覺 SSL | DINOv2 | DINOv3 | ✅ | 同角色升級 |
| 語意/語言對齊 | CLIP | — | ❌ 拿掉 | 失語意 grounding |
| 分割/邊界結構 | SAM | MedSAM2 | ✅ | 醫療版更貼題 |
| **領域特化(視網膜)** | — | RETFound | ➕ 加的 | 楊瀚博沒 domain 教師 |
| 重建目標 | MIM | MIM | ✅ | 一致 |

**關鍵**：唯一功能不一致的換是 **CLIP → RETFound**（拿語意廣度換 OCT domain 深度，對 OCT 任務是升級）。可考慮再加：**OCTCube 當第 4 教師**（蒸 3D OCT 知識）/ BiomedCLIP（補回語意）/ OCT 層分割教師。

---

## Encoder 存亡實驗表（上機填）
每列一個 encoder 變體，跑同一套 Phase 2（凍結 → 同 predictor → 厚度頭），填下游指標。

| # | Encoder 變體 | 厚度 MAE(µm)↓ | change-cond MAE↓ | P0: 會變眼贏 persistence? | L: 贏直接回歸? | 參數/算力 | 結論(留/砍) |
|---|---|---|---|---|---|---|---|
| B0 | **OCTCube**（baseline/閘門） |  |  |  |  |  |  |
| B1 | RETFound-only |  |  |  |  |  |  |
| B2 | DINOv3-only |  |  |  |  |  |  |
| B3 | MedSAM2-only |  |  |  |  |  |  |
| M1 | 多教師（無 JEPA，MIM-only） |  |  |  |  |  |  |
| M2 | 多教師 + 空間 JEPA |  |  |  |  |  |  |
| M3 | 多教師 + JEPA + **厚版 adapter** |  |  |  |  |  |  |
| ★ | **student_full（本方法）** |  |  |  |  |  |  |

**判讀**
- `star > B0`？→ 自建 encoder 才有理由（證成 #1）。
- `M1 > B1/B2/B3`？→ 多教師 > 單教師（證成 #2，Phase1 主結果）。
- `M2 > M1`？→ 空間 JEPA 有用（證成 #4）。
- `M3 > M2`？→ 薄 adapter 確實不夠、厚版才夠（證成 #3）。
- 任一元件沒讓下游變好 → **砍掉**（別為複雜度而複雜度）。

---

## Baseline 群（實作在 `forecast_c/model/baselines.py`）
| 層 | baseline | 實作 |
|---|---|---|
| ① 地板 | persistence | `persistence()` |
| ① 地板 | mean-change | `MeanChangeBaseline` / `fit_mean_change()` |
| ① 地板 | baseline-severity 回歸 | `BaselineSeverityRegressor`（`fit_ols`） |
| ① 地板 | Δt-only | `dt_only_cond()` |
| ① 地板 | treatment-blind | `ForecastModel(treatment=None)` |
| ③ encoder | 換 encoder(OCTCube/RETFound/ImageNet) | `build_encoder_baseline(name)`（L40 接點） |
| ③ encoder | 端到端 3D-CNN（不用 FM） | `EndToEndVolumeRegressor` |
| ④ 方法 | 直接回歸（不經 latent） | `DirectThicknessRegressor` |
| ⑤ 文獻 | anti-VEGF ML / RETFound 下游 / CNN 進展 | `LITERATURE_BASELINES`（引用對比） |

**非贏不可的三個**：② 臨床 tabular GBM（臨床價值）、③ frozen OCTCube（encoder 值不值）、④ 直接回歸（latent 值不值）。

---

## 相關風險 → 對應檢查
| 風險 | 對應消融/檢查 |
|---|---|
| 治療變異不足 → 因果混淆 | A-1 census ⑤（次數-時間 r）；treatment-blind baseline |
| 厚度 ≠ 臨床變乾 | 拿積水標註 / 誠實定位厚度為 proxy（R-1 待醫師） |
| 300 人小樣本 | 病人層級 5-fold CV + 信賴區間 |
| 量測噪音未知 | 同日重掃厚度差 = test-retest 噪音地板（去重前那筆留著算） |
