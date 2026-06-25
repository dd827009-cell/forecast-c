# CLAUDE.md — forecast_c 專案交接（換機/新 session 必讀）

> 這份檔案讓任何機器上的 Claude Code 一打開專案就接得上脈絡。設計細節見 `docs/完整規劃_C.md`、
> 各模組見 `forecast_c/README.md`、跨 session 記憶見 `memory/`。**本檔是單一入口。**

## 0. 一句話定位
**設計 C = 治療條件化 OCT latent 預測**：在凍結的 OCT 表徵上，給「現在 + 治療 + Δt + 病人資訊」
預測「未來 latent」，解碼成**厚度µm + 積水體積變化**。場子 MICCAI / MIA。
用 **JEPA「方法」**（空間+時間兩處），不是 V-JEPA「模型」。

## 1. 架構四階段
- **Phase 1a**（表徵）：2D per-slice 學生 = **多教師蒸餾**（DINOv3 + RETFound + MedSAM2，per-teacher head）+ MIM → 凍結。（楊瀚博 FoundMIM 方法，教師換眼科域）
- **Phase 1b**（跨切片）：**3D adapter**（depth-only 3×1×1 + residual，PEFT）插進凍結 2D 學生，用**空間 JEPA**（遮切片→預測凍結特徵 stop-grad）訓 adapter → 凍結。（吳韋論 Adapter3D + 設計 C 原創 JEPA）
- **Phase 2**（預測）：凍結 encoder → 一步治療條件 predictor（殘差零初始 + change-weighted + 可選 logvar NLL），target = **同一凍結 encoder 編未來 stop-grad（★免 EMA）**。一步 + 短程 rollout。
- **Phase 3**（讀出）：SwinUNETR 解碼 → 厚度 + 積水分割→體積變化。（**尚未實作**，stub）

## 2. 關鍵設計決定（別搞錯）
- ★ **免 EMA**：Phase 2 target 用同一個凍結 encoder 編未來 + detach + LN，不需要 EMA teacher。
- **無 VICReg**：防崩塌靠 stop-grad（+ Phase 1 接地）。
- **殘差零初始**：predictor 起點 = persistence（ẑ=z_t）；adapter 起點恆等。
- **最小版優先**（鐵律）：先「凍結 OCTCube → 一步 predictor → 厚度頭 → vs persistence/直接回歸」出第一個數字，再逐件加（每件過消融閘門）。存亡消融：**P0**（會變的眼上贏 persistence？）、**L**（latent 預測贏直接回歸？）。
- **前輩 code 的定位**：楊瀚博/吳韋論給的是**元件**（蒸餾學生、3D adapter）；**JEPA 訓練法（空間+時間）是設計 C 原創**，兩篇都沒有。

## 3. 現狀（2026-06-24，全在容器 pilot 驗過）
| 元件 | 狀態 |
|---|---|
| OCTCube encoder（凍結，token 網格 5120×1024） | ✅ 真權重 + 真 volume 驗過 |
| 三教師 extractor（DINOv3 / RETFound / MedSAM2） | ✅ 真 B-scan 抽特徵 |
| Phase 1a 蒸餾迴圈（3 教師） | ✅ pilot smoke loss 下降 + 存學生 checkpoint |
| Phase 1b 空間 JEPA（Adapter3D） | ✅ 載 1a 學生 + 訓 adapter，loss 下降 |
| Phase 2 整機（治療條件預測） | ✅ 真 OCTCube latent → predictor → loss |
| 治療 loader（Excel 逐針 → 模型條件） | ✅ 真 EYLEA → cond 77 維 |
| 厚度 GT / A-1 census / h5 讀取器 | ✅ 真 pilot |
| 整合測 | ✅ 15/15 PASS |
| Phase 3（SwinUNETR/積水/REG-1） | ⬜ stub |

## 4. forecast_c 模組地圖
```
config.py            單一真相超參（dataclass，無 torch）
census/              A-1 普查（純 CPU）: cst / recovery / a1_census（--h5-dir 直接讀 h5）
model/               encoder(凍結OCTCube) / treatment / predictor / losses / thickness / forecast_model / baselines
data/                oct_h5(h5讀取+厚度GT+B-scan) / treatment(Excel逐針→TreatmentEncoder dict) / dataset(配對)
phase1/              adapter3d(吳韋論) / foundmim(楊瀚博,vendored) / distill(OCT接合+3教師) /
                     teachers(本地ckpts載DINOv3/RETFound/MedSAM2) / spatial_jepa(★原創) /
                     train_distill(1a迴圈,--save) / train_jepa(1b迴圈,--student-ckpt)
phase3/              SwinUNETR/積水/REG-1 stub（待做）
train/               train_phase2(--smoke) / eval(P0 persistence / L 直接回歸 / change-cond)
tests/run_tests.py   dummy-latent 整合測（15 PASS）
```

## 5. 環境怎麼跑（重要）
- **本機 Windows 原生無 Python**。用 **WSL2 + Docker 映像 `octcube-dev`**（13GB，含 torch/numpy/timm/h5py）。
- 測試/smoke 範例（repo 根目錄下）:
```bash
REPO=/mnt/c/.../forecast-c   # 你的 repo 在 WSL 的路徑
docker run --rm -v $REPO:/workspace -w /workspace octcube-dev python -m forecast_c.tests.run_tests
docker run --rm -v $REPO:/workspace -w /workspace octcube-dev python -m forecast_c.census.a1_census --selftest
docker run --rm -v $REPO:/workspace -w /workspace octcube-dev python -m forecast_c.train.train_phase2 --smoke --steps 5
```
- **MedSAM2 教師需額外裝**：`pip install --no-deps sam2 && pip install hydra-core omegaconf iopath`（已記在 docker/requirements-docker.txt）。
- 容器**無 flash-attn / CPU**：OCTCube 用非 flash 變體（已寫好 key remap）；L40 裝 flash-attn 會更快。

## 6. 新機器 setup 步驟（從零）
1. `git clone <this repo>` → 進 repo 根。
2. **OCTCube 官方碼**：另 clone `OCTCubeM`（含 `OCTCube/models_vit_st_joint.py`）放到 repo 根的 `OCTCubeM-main/`（`forecast_c/model/encoder.py` 預設 `repo_dir="OCTCubeM-main/OCTCube"`）。
3. **權重放 `ckpts/`**（不進 git）：`OCTCube.pth`、`RETFound_mae_natureOCT.pth`、`dinov3_vitl16_pretrain_lvd1689m-*.pth`、`MedSAM2_latest.pt`。下載連結見 §9。
4. **建/拉 docker 映像**：見 `docker/SETUP_WSL2.md` + `docker/Dockerfile`。
5. **資料**：h5_output（stage0 產出的 study .h5）放本機 SSD（不進 git）。
6. 跑 `python -m forecast_c.tests.run_tests` 確認 15/15 PASS。
7. **memory**：把本 repo `memory/*.md` 複製到 `~/.claude/projects/<新專案路徑的key>/memory/`（key = 專案絕對路徑把 `\/:` 換成 `-`）。

## 7. 資料真相 + 連結（見 memory/dataset-ilm-bm.md）
- OCT：h5 schema = `volume(25,496,512)` + `ilm_y/rpe_bm_y(25,512)` + `valid_ascan_mask` + attrs(`longitudinal_key=patient_id::OD/OS`, `acquisition_time_utc`, scales, age, sex)。
- **治療連結鍵 = `病歷號` ↔ h5 `patient_id`**（同 ID 系統）+ `OS/OD` ↔ laterality + 施打日期 ↔ visit date。
- 治療檔：`EYLEA 8mg…xlsx`「重新整理過」sheet（逐藥逐針日期，主訊號）；`全體系case pooling…xlsx`（828 眼，變乾 label + baseline markers，補充）。⚠️ case-pooling 的「Chart no.」是別的序號，**勿用**。
- 變乾 label 現成：EYLEA `Time to Dry` / case-pooling `Dry macula after loading doses`。

## 8. 還缺什麼 / 下一步
**🔴 卡資料/人**：① 治療 cohort 的**真實 OCT 影像**（pilot 81 不是 EYLEA 病人，沒重疊）→ 拿到才能 join 跑 Phase 2 治療版。② 醫師 R-1（變乾閾值/積水標註）。
**🟡 不卡、可做**：
- **教師特徵預計算腳本**（L40 訓練前置；楊瀚博式存 SSD，現在這台就能跑）→ 見 §10 避坑。
- **Phase 3**（SwinUNETR + 積水 + REG-1）。
- 把 `data/treatment.py` 接進配對 dataloader。
- ⚠️ **`train_distill` 要把 `reconstruct_orig_img=False`**（對齊楊瀚博 proposed，現在是 True 會被像素 MAE 主導）。

## 9. 權重下載（放 ckpts/，永不連 HF）
- OCTCube：`OCTCubeM` repo 的 ckpt 連結（OCTCube.pth = SSL base，**非** multitask_cls）。
- RETFound OCT（gated）：`huggingface.co/YukunZhou/RETFound_mae_natureOCT`。
- DINOv3 ViT-L/16（gated）：`facebook/dinov3-vitl16-pretrain-lvd1689m`（官方 checkpoint，teachers.py 有 key remap）。
- MedSAM2：`huggingface.co/wanglab/MedSAM2` → `MedSAM2_latest.pt`。

## 10. 楊瀚博避坑清單（教師特徵預計算 + 蒸餾，務必看）
1. **前處理「預計算」與「訓練」逐像素一致**；**不做資料增強**（教師特徵是固定靶）。
2. **教師特徵存原生解析度**（DINOv3 14×14×1024 / MedSAM2 64×64×256），對齊在 translator bilinear 做。
3. **per-channel 正規化**（用 train split 全域 mean/var），否則尺度大的教師主導 loss。
4. **`reconstruct_orig_img=False`**（proposed 只做教師蒸餾，不做像素 MAE）。
5. **per-teacher 4 層 MLP head**（1 層不穩、掉 4%）；**mask ratio 0.5**。
6. 超參：AdamW、有效 batch **1024**（單卡 = batch 64 × accum 16）、lr **1.5e-4** + cosine + warmup 10、mixed precision（⚠️ + cosine normalize 易 NaN，加 eps）。
7. **特徵必須放 SSD**（HDD/NAS → I/O 卡死「好幾週」）；**打包成 shard** 避免幾百萬小檔爆 inode；預計算要 **resumable** + 記錄版本（改前處理→預存失效）。
8. **多尺度限制**：只在最後一層蒸餾 → DPT/SwinUNETR 解碼器會弱 → **Phase 3 要提前規劃階層式蒸餾 or ViT-Adapter**。
9. **洩漏**：病人層級切分**連 Phase 1 都要套**（test 病人的眼不進 pretraining）。
10. **OCT 域特有**：灰階→3ch；B-scan 496×512 非正方 resize 會壓扁（做消融）；DINOv3/MedSAM2 沒看過 OCT（先 linear probe sanity check）；壞 B-scan 用 `image_quality_per_bscan` 濾掉。

## 11. 設備（前輩都用單張 L40 48GB）
- 楊瀚博 ViT-B：1× L40 + 515GB RAM，batch 64×accum16，100 epochs，**~2 天**（教師特徵預先存 SSD）。
- 吳韋論：1× L40，PEFT（只訓 adapter），batch 8，200 epochs。
- → 設計 C Phase 1 **單張 L40 就能跑完**；目前若只有小卡（如 3060Ti 8GB）→ 做資料整理/census/教師特徵預計算/smoke，全量訓練等 L40。
