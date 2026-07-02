# Memory Index（forecast-c 新 repo，2026-06-24 整理）

> 已移除被設計 C 覆寫的舊記憶（repo-build-state=V1 / stage-a-redesign=Stage A / authoritative-roadmap=舊文件）。
> 專案完整交接見 repo 根 `CLAUDE.md`。

- [★現行設計 C](design-c-current.md) — 2D 多教師蒸餾(DINOv3/RETFound/MedSAM2)+ 3D 聚合(空間JEPA)+ 治療條件時間預測 → 厚度+積水；白紙重寫成 `forecast_c/`，最小版優先、免EMA、官方code照用；容器測 15/15 PASS。
- [前輩可重用 code](predecessor-code.md) — 楊瀚博 FoundMIM(蒸餾+MIM)→Phase1a；吳韋論 Adapter3D(depth-only 3×1×1)→Phase1b；**已 vendored 進 forecast_c/phase1**；OCTCube/三教師真權重接好、Phase1a/1b/2 迴圈 pilot 跑通；JEPA 訓練法是設計 C 原創。
- [資料真相+治療連結](dataset-ilm-bm.md) — h5 schema + **連結鍵=病歷號↔patient_id**；EYLEA 檔逐針日期(主訊號)+ case-pooling(變乾label)；治療 loader 已建、cond 77維驗通；治療 cohort 真實 OCT 待補。
- [Excel join 欄位實測](excel-join-keys-verified.md) — EYLEA 病歷號(A) + Pool Chart no.(D) 都是 7–8 位醫院病歷號(可 join)；**推翻**舊「pool Chart no. 5碼勿用」；pool Name 此版是真姓名。配 `pipeline/`（NAS 先掃再取 + 每 .pat log 病歷號+.sdb 數）。
- [可選方向](optional-directions.md) — I-1 治療反應分型(latent軌跡聚類) + I-2 擴散模型(質化渲染器)；未拍板。
- [本機環境](no-python-on-windows-box.md) — Windows 原生無Python，WSL2+Docker `octcube-dev` 跑；訓 backbone 須 L40（換機後依新機調整）。
