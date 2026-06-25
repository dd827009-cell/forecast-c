# Vendored code — provenance

本目錄 = **vendored 楊瀚博碩論 FoundMIM**（多教師蒸餾 + MIM）原始碼。

- 來源: `C:\Users\Administrator\Desktop\楊瀚博\program\pretrain\mim_modeling\`
- 論文: 楊瀚博,《Foundation Model in Colonoscopy Image Using Multi-teacher Distillation with Masked Image Modeling》(NTU, 2025)。
- 檔案: `models_mim.py`（主模型 MaskedImageModelingViT）、`models_mim_sak.py`（SAK 變體）、
  `cross_attn.py`、`adapter_heads.py`、`feature_translators.py`（Theia 式）、`pos_embed.py`。
- 原碼進一步參考: MAE / MILAN / DMAE / EfficientSAM / AM-RADIO / Theia（見各檔頂 header）。

**未改動**原始實作（保持可對照）。OCT 域接合（教師換 DINOv2/RETFound/MedSAM2、輸入域）在
上層 `forecast_c/phase1/distill.py`，不動本目錄。
