---
name: design-c-current
description: 最新拍板設計 = 設計 C(取代 V1/V2/A/B):2D 多教師蒸餾學生 + 3D 聚合(JEPA)+ 治療條件時間預測 → 厚度+積水體積變化。用 JEPA『方法』非 V-JEPA『模型』。資料夾已重整、目前手上無資料。
metadata: 
  node_type: memory
  type: project
  originSessionId: 12c34794-e6f2-434e-92d9-ba9781bf7c96
---

> 2026-06 大重整。**這是現行設計,覆寫 [[stage-a-redesign]](=設計A,已封存)與 [[repo-build-state]] 的 V1。**

> **⚡ 2026-06-24 白紙重寫**:前幾代程式(`latent_dynamics/ backbone/ peft/ multitask/ stage1/ train/`)全封存到 `_archive/C_pre_rewrite/`,開全新乾淨 package **`forecast_c/`**(2168 行,25 檔,容器 octcube-dev 全測 PASS 14/14)。**最小版優先**:凍結 OCTCube → 一步治療條件 predictor(殘差零初始+change-weighted+可選 logvar NLL)→ 厚度頭 → vs persistence+直接回歸 + **A-1 普查(純CPU,唯一現在可跑)**。**★關鍵修正:免 EMA**(target=同一凍結 encoder 編未來 stop-grad,刪掉 EMATeacher)、無 VICReg、LN 空間一致。Phase1 多教師蒸餾(楊瀚博法,DINOv2/RETFound/MedSAM2)+ 3D adapter(吳韋論 PEFT 概念)、Phase3 SwinUNETR(MONAI)只留 `phase1/ phase3/` 介面 stub。原則:**有官方代碼就用官方代碼**(encoder 接 OCTCubeM-main 官方 MAE;教師/SwinUNETR 待接)。L40 接點(`build_octcube_encoder`/`build_dataloader`)NotImplementedError 清楚標。兩篇論文無公開 code→照方法用官方教師自組。**未 commit**(交使用者決定)。

## 設計世代:V1(world model)→ V2(雙軌)→ A(Stage A)→ B(OCTCube 蒸餾)→ **C(現行)**

## 設計 C 架構(方案①:2D 蒸餾 + 3D 聚合)
- **Phase 1a**:2D per-slice 學生 = **多教師蒸餾(DINOv2 + RETFound + MedSAM2,per-teacher heads,2D→2D)+ 2D MIM + 2.5D 鄰窗** → 凍結。(≈ 楊瀚博,教師換領域)
- **Phase 1b(跨切片=B,已拍板)**:**3D adapter 插進凍結 2D 學生**(**depth-only 3×1×1 + residual**,PEFT,base凍只訓 adapter,吳韋論式;非獨立聚合、非 2.5D)→ 用 **空間 JEPA(訓 adapter)+ ILM/BM 厚度監督 + 時間相干性(縱向夠才加)** 訓 → 凍結。穿透每層、不洗掉、**3×1×1 保 in-slice**;adapter on/off = 天然消融。
- **凍結前閘門**:Track A 探測 **vs OCTCube**,贏才凍結。
- **Phase 2**:凍結 encoder → 治療條件 **時間 JEPA** 預測(z_t +[藥身份,次數,Δt,年齡,性別]→ẑ,殘差+單尺度條件+change-weighted,stop-grad target 免 EMA,一步+短程 rollout)。
- **Phase 3**:SwinUNETR 解碼 → 厚度µm + **積水分割→體積變化(這次−上次)**;真實 25 切片 slab @512 + REG-1 配準。
- **🔑 關鍵澄清**:用 **JEPA「方法」(空間+時間兩處),不是 V-JEPA「模型」**。② V-JEPA 模型僅後備(①3D太弱/reviewer 要 joint-3D 對照)。

## 任務 / 資料
- 預測**治療後厚度 + 積水**(積水=每次分割算體積,看 this−last 減少量;非存活/CST 代理)。
- ~600 病人 / ~1200 眼 / 3-5 次 / 單中心 / 每張有 ILM/BM(厚度真 GT)/ **積水無標註**(MedSAM2 蒸餾 + 自標 50-100 張校準)。
- 治療只有藥身份+次數(無劑量);病人=年齡/性別。
- **⚠️ 使用者目前手上沒資料 → 卡資料;拿到後第一步 = A-1 普查腳本(規格 `solutions/A1_census_spec.md`,純 CPU)。**

## 存亡風險(最先驗的兩個消融)
- **L**:latent 預測可能輸給「直接回歸」→ **核心消融,輸了重新定位**(賣標籤效率/多步/遷移)。
- **K**:空間 JEPA 可能只在 trivial 內插(切片稀疏)→ on/off 消融,沒比厚度-only 好就砍。
- 其他:M 積水分割跨回診一致性 / N off-manifold 解碼 / O 解碼器多尺度來源(優先階層式學生,非硬接 adapter)。
- **鐵律**:先做最小版(2D學生+一步預測+厚度頭)出第一個數字,再逐件加+消融。

## 定位
**整合級貢獻 → MICCAI/MIA**(非純方法刊);world model 只當血緣。「方法導向」天花板=方法味醫學影像,跟楊瀚博同級。

## 資料夾(2026-06-24 重寫後)
- **`forecast_c/`** — ★**現行程式**(白紙重寫,最小版):config/census/model/data/train/phase1/phase3/tests + README。容器跑法見 README。
- `latent_forecast_C/` — **設計文件**:`完整規劃_C.md` + `待辦.md`(含最小驗證順序);pptx 畫圖腳本移 `figures/`。
- `solutions/` — **共用資產**:Pipeline(前處理)/ A1_census_spec / EVALUATION / RELATED_WORK / RUNBOOK_NAS / 方法論文_Q1規劃。
- 資料管線(保留未動):`stage0/`(m1-m7b)`test1/ pdb_to_h5/ h5_output/ configs/ docker/ scripts/`。
- `_archive/` — `V1V2/ A_StageA/ B_OCTCube/` + **`C_pre_rewrite/`**(latent_dynamics/backbone/peft/multitask/stage1/train,前幾代設計層,有 _README)。
- 環境:Windows 原生無 Python,跑測試用 **WSL→docker `octcube-dev`**(`docker run --rm -v /mnt/c/.../pretrain:/workspace -w /workspace octcube-dev python -m forecast_c.tests.run_tests`)。
- ⚠️ root 6 個已追蹤 V1 檔已移 `_archive/V1V2/`(git working tree 顯示 D,可逆,GitHub 未受影響直到 push)。

相關:[[stage-a-redesign]](A,已封存)、[[dataset-ilm-bm]]、[[repo-build-state]](V1,已過時)、[[no-python-on-windows-box]]。
