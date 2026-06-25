---
name: dataset-ilm-bm
description: "資料真相:~20+TB OCT 大檔案庫 + 小精選子集(19病人1.58GB,含治療過程+ILM/BM)。每study=25張OCT B-scan+1張IR。→ 20TB 餵 Phase1 續訓、子集餵 Phase2;厚度頭可用 BM−ILM 真監督;變乾定義待醫師(R-1)。"
metadata: 
  node_type: memory
  type: project
  originSessionId: a552fdf0-9f5c-4e6b-85fa-6bf05c68f6d2
---

使用者 2026-06-16 說明資料**真相**（**推翻先前「完全沒資料集」**）：
- **大檔案庫 ~20+ TB OCT**：**每張影像都有 ILM/BM 邊界 + 同一機型**；但**不一定都是縱向+有治療**。
- **小精選子集**：19 病人 = 1.58GB（≈85MB/病人），有完整治療過程。
- **每 study 格式**：從 `.pdb` 解出 **25 張 OCT B-scan + 1 張 IR(en-face/眼底)**。25 層=稀疏黃斑 cube，OCTCube(3D)可用但要統一重採樣深度。IR 可走 OCTCube-M 多模態。
- 粗估 20TB ÷ 85MB ≈ **十萬量級 study**，與 OCTCube 自身語料同級或更大（但 raw TB≠可用量，**第0步必先普查**）。

**ILM/BM 是密集逐 A-scan**（schema 有 `ilm_y, rpe_bm_y` 列座標 + `valid_ascan_mask`）→ L_layer 做**密集厚度圖**(最強)，非純量。厚度 µm=(rpe_bm_y−ilm_y)×3.87167。

## 資料三層（決定每張能餵什麼）
1. **全部**(有 ILM/BM+同機型) → MAE 重建 + **層/厚度監督頭(L_layer)**。
2. **縱向子集** → ＋時間相干性 / 時間遮罩。
3. **縱向+治療子集**(最小) → Phase 2 Stage A。

## 預訓練策略（結論：continual/DAPT，不 from-scratch、不只凍原始）
- **Phase 1 續訓 = 從 OCTCube 續訓**(白拿知識)。from-scratch=陷阱(資源少必更差);只凍原始=浪費20TB但仍是**必比的 baseline**。
- **★只做 MAE 會浪費「全資料都有 ILM/BM」這個標籤** → 改**多任務持續預訓練**：`L_recon`(MAE,全資料) + `L_layer`(ILM/BM 邊界/厚度監督,全資料,方向同下游=白賺) + 時間項(縱向子集)。Kendall 配重。紀律:MAE 仍主角、層頭當 auxiliary、probe 閘門、別變純分割。
- **時間遮罩(VideoMAE/V-JEPA tube masking)出局**：每眼只 **3-5 次回診**(稀疏+不規則),對影片式遮罩太少(VideoMAE 要幾十~上百幀);高遮罩率下=「用1-2次補3-4次」≈Stage A 預測本身,且這序列價值在 Stage A 監督、不該燒在 SSL。→ Phase 1 時間訊號改**成對(pairwise)相干性**:Δt 預測(5次=10對)/順序-進展軸 LSSL(靠多眼撐),在對/序列層級運作、不需密集幀。
- **同機型**=預訓練乾淨但 encoder 沒見過別機型 → 換機型外部驗證會弱,論文 limitation 誠實寫。
- **Phase 2 Stage A = 需縱向+治療+ILM/BM → 只用精選子集**，凍住「適配後」encoder。
- 防遺忘:低LR+LoRA或EMA;每階段 probe 閘門(贏凍原始才留);先小批 pilot 再砸全量(算力見 [[no-python-on-windows-box]] L40)。
- 階段:0普查→1凍原始baseline→2續訓MAE(+時間相干性)+reprobe→3 Stage A。

## 厚度頭設計影響
- **厚度頭升級為真監督**：`視網膜總厚度 = BM − ILM` 是 ground-truth，不再用分割算的 CST 代理。可做中央子場/ETDRS 九宮格/整張厚度圖回歸。
- 若 ILM/BM 是**逐 A-scan/B-scan 邊界**（非單一純量）→ 可做**密集厚度預測**，正好用上 MAE 密集細節 → 再次坐實 [[stage-a-redesign]] 的「MAE 優於 I-JEPA」。

## 🆕 2026-06-24 治療紀錄 Excel（解掉治療 metadata 卡點）
檔案:`C:\Users\Administrator\Downloads\全體系case pooling (20250811更新) (2).xlsx`，sheet `data collection` = **828 列 × 73 欄**（nAMD/PCV/RAP 抗 VEGF 病例池）。關鍵欄:
- **治療**:3 種 anti-VEGF(Eylea/Lucentis/Avastin) → **藥種有變異**(Stage B 換藥或許可試);Regimen(TE/PRN/fixed/mixed);**注射次數**(0–12mo、12–24mo 兩窗) → ⚠️是**窗口彙總數非逐次日期**(治療條件 granularity 較粗)。
- **★變乾 label 現成**:`Dry macula after loading doses`(0/1) + `Fluid-free interval (months)` → 存活頭/變乾**免從 CST 推**(補強 recovery.py;醫師仍要確認定義)。
- baseline 積水標記:IRF/SRF/SHRM/Subretinal fibrosis/PED>400µm;診斷亞型;VA(baseline/6/12/18/24mo);age/sex/eye(OD/OS);apply date/last visit/follow-up。
- ✅ **連結鍵=`病歷號`(2026-06-24 更正)**:新檔 `EYLEA 8mg 恩慈整理完成 (3).xlsx`「重新整理過」sheet 的 `病歷號`(如 20592156/21396405)與 h5 `patient_id`(如 20242932/21394651)**同 ID 系統**(同 6–8 碼、號段交錯)→ **join=(病歷號, OS/OD eye, 施打日期↔visit date)**。之前 case-pooling「Chart no.」(10160 那種 5 碼)是**別的序號,勿用**。pilot 17 病人 ∩ EYLEA=0 只因 pilot 非 EYLEA cohort(待真實 OCT 重疊確認 patient_id==病歷號)。
- **EYLEA 檔 = 細粒度治療(模型主訊號)**:逐藥逐針日期(Aflibercept_8mg/2mg、Bevacizumab=Avastin、Ranibizumab=Lucentis、Faricimab=Vabysmo、Brolucizumab=Beovu…)+ Naïve + CRT/IRF/SRF + Time-to-Dry。case-pooling=補充(828眼、變乾label、baseline markers,但治療只到窗口針數)。
- ✅ **治療 loader 建好**:`forecast_c/data/treatment.py`(`parse_treatment`→178條軌跡;`treatment_dict_at(visit_date)`→drug_ids/numerics[距針天數,序,累積]/is_naive)。**真 EYLEA → TreatmentEncoder → cond(77維) 驗通**。差真實 OCT 影像 join。
- pilot 結構(81 檔):**32 眼 / 17 病人 / max 12 visits/眼 / 10 眼 ≥2 visit** → 縱向邏輯可在 pilot 驗。A-1 census 已可直接讀 h5(`a1_census --h5-dir h5_output`,免 manifest);跑出決策表(暫定 300µm 閾值下變乾率高、median 1 visit、會變眼 50%)。
- 權重也到位:**OCTCube 權重 = `OCTCubeM-main/OCTCube.pth`**(+ `OCTCube_multitask_cls.pth`),非只 ckpt.txt。

## ⚠️ 未決（醫師 R-1 拍板）
- ILM−BM = **總厚度**；「變乾」臨床通常指**積液消失**(intraretinal/subretinal fluid)。厚度高≠有積液(可能纖維化)。
- 「變乾」定義 = 厚度<閾值 還是 積液量<閾值？若資料**另有積液標註**，存活頭標籤優先用積液，厚度當輔助/平滑目標。
- 關聯：[[stage-a-redesign]] 的 C-1 變乾重定義 / recovery.py 閾值。

## 待辦（使用者已 greenlight 厚度頭，未拍板要不要現在動工）
- 把厚度頭規格從「CST 代理」改成「BM−ILM 真監督(純量/厚度圖)」寫進 solutions/MODEL_DESIGN.md + EVALUATION.md。
- dummy 補厚度頭 + 厚度損失(已有層不交叉/平滑約束 F-1/F-2)骨架。
