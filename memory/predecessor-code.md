---
name: predecessor-code
description: 楊瀚博(多教師蒸餾+MIM)與吳韋論(3D adapter+PEFT)的「完整可跑 code」位置與對 design C 的可重用對應。
metadata: 
  node_type: memory
  type: reference
  originSessionId: f15d1d9e-94f3-4503-aee9-18438eee8c8a
---

> 2026-06-24 發現:兩位前輩的**完整實作 code**(非只論文)在桌面,可直接重用接進 [[design-c-current]] 的 `forecast_c/phase1`。
> **✅ 2026-06-24 已 vendored 進 `forecast_c/phase1/`(容器測過):** `adapter3d.py`(吳韋論 Adapter3D,參數化+殘差零初始起點恆等+inject 凍結學生)、`foundmim/`(楊瀚博 FoundMIM 整個 mim_modeling 原樣複製,有 NOTICE)、`distill.py`(OCT 域 wrapper:build_foundmim_for_oct+教師 spec DINOv2/RETFound/MedSAM2+extractor stub)、`spatial_jepa.py`(★原創:遮切片→adapter 學生預測被遮切片表徵,target=凍結 base stop-grad,只訓 adapter)。RETFound/MedSAM2 教師 extractor 待接官方權重(DINOv2 可接 timm)。

## 楊瀚博 — FoundMIM(多教師蒸餾 + MIM)→ Phase 1a
路徑:`C:\Users\Administrator\Desktop\楊瀚博\program\pretrain\`
- `mim_modeling/models_mim.py` = **主模型** `MaskedImageModelingViT`(打磨完整):任意 teacher_names + feature_sizes、cross-attn/self-attn teacher decoder、cos_sim/l2 loss、**自適應 per-patch 教師加權 + load-balance**、direct-distillation 選項、原始 MAE recon。`forward(imgs, mask_ratio, teacher_features={})`。
- `mim_modeling/`:`adapter_heads.py`(LightConvAdapterHead)、`feature_translators.py`(Theia 式)、`cross_attn.py`(MIM decoder)、`pos_embed.py`。
- `vfm_feature_extraction.py` + `foundation_models/`(clip/dinov2/sam/vit)= 教師特徵預計算。`dataset.py`(PretrainDataset 載預存特徵 + FeatureNormalizer)。`foundmim_engine_pretrain.py`/`_main_`/`.sh`。
- `models_mim_sak.py` = SAK(教師專屬內部路徑,論文未來方向)。
- **改造**:教師換 DINOv2(留)+RETFound+MedSAM2(替 CLIP/SAM,**缺這兩個 extractor**);輸入域 colonoscopy 224³ch → OCT B-scan(灰階→3ch)。學生 ViT-S/B,凍結後接 Phase 1b。

## 吳韋論 — USFM + PEFT(adapter3D+DoRA+dualbranch)→ Phase 1b 跨切片
路徑:`C:\Users\Administrator\Desktop\吳韋論\program-wl\`
- `models/utils/peft.py` = **`Adapter3D`**:`Conv3d(kernel=(3,1,1),padding=same)` + linear down/up bottleneck + residual + LN = **depth-only 3×1×1**,**正好是 design C §4 要的跨切片機制**(吳韋論消融最強元件)。reshape (B,N,C)→(B/d_size, d_size, h, w, C) 做 3D conv。也有 AdaptFormer/Convpass。
- `models/utils/dora.py`(DoRA)、`pissa.py`(PiSSA)。`models/*_3d.py`(vit_3d/swin_3d/convnextv2_3d…)。base = **USFM**(`pretrain/USFM_latest.pth`,凍結 `freeze:true`)。
- proposed config `config/proposed_BM_dualbranch_adapter3d_dora.json`:peft_type=adapter3D, adapter_size=64, d_size=16, lora_type=dora, lora_r=8 + ViT-Adapter(deform/interaction_indexes)。
- **改造/差異**:吳韋論用在**分類微調**;design C 要把 Adapter3D 插進凍結 2D 學生 + 用**空間 JEPA SSL** 訓(這部分兩篇都沒有)。

## ✅ 2026-06-24 真權重接上 + pilot 驗證（容器 octcube-dev，CPU）
權重全在 `ckpts/`(DINOv3-L 1.2G/RETFound-OCT 3.7G/MedSAM2 149M/OCTCube.pth 3.98G)。h5_output 有 **81 筆 pilot**(schema: `volume(25,496,512)`、`ilm_y/rpe_bm_y(25,512)`、`ir(768²)`、attrs 含 `longitudinal_key=patient_id::OD/OS`、scales、fovea)。
- **教師 extractor** = `forecast_c/phase1/teachers.py`(吃本地 ckpts，離線):RETFound→timm `vit_large_patch16` 直接載;DINOv3→timm `vit_large_patch16_dinov3` + **鍵 remap**(`storage_tokens→reg_token`、`ls1/ls2.gamma→gamma_1/2`、丟 `qkv.bias_mask`/`rope_embed.periods`)。兩者真 B-scan 抽 patch 特徵 **[B,196,1024]** 驗過(灰階→3ch→224→ImageNet norm)。MedSAM2=SAM2 Hiera 需 `sam2` 套件 → 待接。
- **OCTCube encoder** = `forecast_c/model/encoder.py:build_octcube_encoder`:用官方非flash `models_vit_st_joint.vit_large_patch16`(num_frames=60,t_patch=3,img=256,high_res_img=512,sep_pos_embed,cls_embed)。**OCTCube.pth 是 flash 命名 → `_octcube_remap` 把 `mixer.Wqkv`(3072)拆 `attn.q/k/v` + `out_proj→proj`**，載入 0 unexpected。**256×256×60frames → token 網格 (5120,1024) = 正好對 BackboneSpec**。`OCTCubeTokenEncoder.prep`(vol→trilinear→min-max)。CPU 載22s+forward16s(L40 裝 flash 更快)。
- **★端到端打通**:真 pilot volume → OCTCube latent → ForecastModel(predictor+厚度) → loss，跑通(z_hat 5120×1024)。最小版 Phase2 已能吃真 OCTCube 特徵。整合測 15/15 PASS。

## ✅ 2026-06-24 Phase 1 訓練迴圈跑通（pilot，治療無關）
治療連結卡住(Excel↔h5 ID 不合)→ 使用者拍板**先跳過治療做 Phase 1**。
- `forecast_c/data/oct_h5.py` = h5 讀取器(volume/B-scan/**厚度GT (rpe-ilm)·axial→(25,512)µm**/CST)，真 pilot 驗過(194–345µm)。
- `forecast_c/phase1/train_distill.py` = **Phase 1a 蒸餾迴圈(三教師齊)**:B-scan→DINOv3+RETFound+**MedSAM2**特徵→FoundMIM forward→cosine+MAE loss→訓學生→**存 2D 學生(--save / smoke 存 ckpts/phase1a_student_smoke.pth)**。pilot smoke 三教師 loss **7.71→5.46 下降**。
- **MedSAM2 教師** = `teachers.py`:`sam2.build_sam(configs/sam2.1/sam2.1_hiera_t.yaml, ckpts/MedSAM2_latest.pt).image_encoder` → 1024 輸入 → `vision_features(B,256,64,64)`→`[B,4096,256]`。需 `pip install --no-deps sam2 && pip install hydra-core omegaconf iopath`(已記進 docker/requirements-docker.txt)。MedSAM2=Hiera-tiny。
- `forecast_c/phase1/train_jepa.py` = **Phase 1b 空間 JEPA 迴圈**(★原創):**載 Phase 1a 學生 checkpoint(--student-ckpt)** + per-slice embed→SpatialJEPA(遮切片→Adapter3D 從鄰切片預測凍結特徵)→只訓 adapter+mask_token。pilot smoke 載入 1a 學生後 loss **0.80→0.65**、base 無梯度。**1a→1b 已串接**。
- A-1 census 可直接讀 h5(`a1_census --h5-dir h5_output`)。整合測 15/15 PASS。
- **L40 待辦**:教師特徵預計算存檔(省 on-the-fly)、裝 flash-attn(OCTCube/速度)、真規模訓練、治療連結(等使用者給 chart↔patient_id 對應)、Phase 3(SwinUNETR+積水+REG-1)。

## ⚠️ 關鍵 gap:JEPA 訓練法兩篇都沒有
前輩給的是**元件**(蒸餾學生、3D adapter),**不是訓練方法**。design C 的「空間 JEPA(遮切片→預測凍結特徵 stop-grad)+ 時間 JEPA(治療條件預測)」是**原創**,要自己寫 = 主要剩餘新穎工作。楊瀚博=蒸餾+MIM(非JEPA)、吳韋論=分類微調(非SSL)。
