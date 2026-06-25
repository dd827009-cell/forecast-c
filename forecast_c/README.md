# forecast_c — 設計 C「治療條件化 OCT latent 預測」（最小版，白紙重寫）

> 規劃見 `latent_forecast_C/完整規劃_C.md`；前幾代封存於 `_archive/C_pre_rewrite/`。

**最小版** = 凍結 OCTCube → 一步治療條件 predictor（殘差+change-weighted）→ 厚度頭 → vs persistence + 直接回歸。
解耦「測預測」（用現成 OCTCube）和「造 FM」（蒸餾，Phase 1 之後加）。

## 結構
```
config.py            單一真相超參（純 dataclass，無 torch）
census/              ★ A-1 普查（純 CPU，唯一現在可跑）: cst / recovery / a1_census
model/               encoder(凍結) / treatment / predictor / losses / thickness / forecast_model / baselines
data/                配對 dataset（dummy + L40 stub）
train/               train_phase2（--smoke 可跑）/ eval（P0 persistence skill, L 直接回歸, change-cond）
phase1/              ✅ 表徵階段:adapter3d(vendored 吳韋論)/foundmim(vendored 楊瀚博)/distill(OCT接合)/spatial_jepa(★原創)
phase3/              之後階段介面 stub（SwinUNETR 解碼 + 積水體積 + REG-1）
tests/run_tests.py   dummy-latent 整合測（15 PASS）
```

## 關鍵設計（與前幾代差異）
- ★ **免 EMA**: target 由**同一個凍結 encoder** 編未來 + stop-grad（不需 EMA teacher）。
- 無 VICReg；殘差零初始（persistence 起點）；z_t 與 target 同在 LN 空間；change-weighted loss。
- 一步預測（多步 rollout / 存活頭待 A-1 普查確認 ≥3 visit 才加）。

## 跑法（容器 `octcube-dev`）
```bash
# 整合測（全 PASS）
docker run --rm -v "$(pwd)":/workspace -w /workspace octcube-dev python -m forecast_c.tests.run_tests
# A-1 普查 self-test（純 CPU）
python -m forecast_c.census.a1_census --selftest
# 訓練迴圈 smoke（dummy encoder + loader，★無 EMA）
python -m forecast_c.train.train_phase2 --smoke --steps 5
```

## 已接好的前輩 code（phase1）
- `phase1/adapter3d.py` — vendored 吳韋論 Adapter3D（depth-only 3×1×1，跨切片，起點恆等，inject 凍結學生）
- `phase1/foundmim/` — vendored 楊瀚博 FoundMIM（多教師蒸餾+MIM 主模型，未改動，見 NOTICE.md）
- `phase1/distill.py` — OCT 域接合（教師 DINOv2/RETFound/MedSAM2 + build_foundmim_for_oct）
- `phase1/spatial_jepa.py` — ★ 空間 JEPA 訓 adapter（遮切片→預測凍結特徵 stop-grad，**設計 C 原創**）

## 卡 L40 / 資料 / 權重的接點（NotImplementedError，清楚標 TODO）
- `model/encoder.build_octcube_encoder` — OCTCube 權重在 `OCTCubeM-main/OCTCube.pth`，需 token 網格 forward + flash-attn
- `data/dataset.build_dataloader` — 縱向配對 shard（病人層級 split 防洩漏）+ 治療 metadata + 厚度 GT
- `phase1/distill.TeacherFeatureExtractor` — RETFound/MedSAM2 教師權重+repo（DINOv2 可接 timm）
- `phase3/`（SwinUNETR + 積水體積 + REG-1）

## 拿到資料後第一步
`python -m forecast_c.census.a1_census --manifest stage0/manifest.parquet --m7b-dir <dir> --pilot 81 --out census_out/`
→ `census_report.md` 決策表決定: 存活頭? 一步/多步? Stage B 可不可? persistence 招牌立不立得住?
