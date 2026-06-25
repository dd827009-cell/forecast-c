"""Phase 1a — 多教師蒸餾 + MIM（OCT 域 wrapper）。

底層模型 = **vendored 楊瀚博 FoundMIM**（`forecast_c/phase1/foundmim/`，原碼來自
  `Desktop/楊瀚博/program/pretrain/mim_modeling/`，碩論《多教師蒸餾 + MIM 腸道 FM》）。
本檔只做 **OCT 域接合**：教師換成眼科（DINOv2 + RETFound + MedSAM2）、輸入域改 OCT B-scan。

底層 `MaskedImageModelingViT.forward(imgs, mask_ratio, teacher_features)`:
  - 遮 mask_ratio → ViT 編可見 → (cross-attn) decoder 補 → feature_translator 轉各教師格式
    → cos_sim/l2 loss 對教師（教師特徵**預計算**，這裡吃 dict）。
  - teacher_features[name] 形狀 = **[B, H*W, C]**（C,H,W = feature_sizes_teacher[name]）。
  - 另含原始 MAE 像素重建 loss（reconstruct_orig_img）+ 自適應 per-patch 教師加權（選用）。

教師（凍結 + 特徵預計算）官方來源:
  - DINOv2 : timm / torch.hub（'vit_large_patch14_dinov2'）——可直接接。
  - RETFound: 官方 repo（rmaphoh/RETFound_MAE）+ 權重 —— extractor 待接。
  - MedSAM2: 官方 repo（bowang-lab/MedSAM2）+ 權重 —— extractor 待接。
⚠️ B-scan 灰階 → 複製成 3ch 餵教師（教師皆 RGB 預訓）。學生 in_chans 預設 3（複製）。
"""
import torch

from .foundmim.models_mim import (mim_vit_small_patch16, mim_vit_base_patch16,
                                  mim_vit_large_patch16)

STUDENT_FACTORY = {
    "small": mim_vit_small_patch16,    # ViT-S 22M
    "base": mim_vit_base_patch16,      # ViT-B 86M（楊瀚博主力）
    "large": mim_vit_large_patch16,
}

# 教師特徵尺寸 [C, H, W]（224 輸入，實測對齊 teachers.py 載入的真模型）。
OCT_TEACHERS = {
    "dinov3_vitl16":  [1024, 14, 14],   # DINOv3 ViT-L/16 @224 → 14×14（patch16）
    "retfound_oct":   [1024, 14, 14],   # RETFound MAE ViT-L/16 @224 → 14×14
    "medsam2":        [256, 64, 64],    # MedSAM2 = SAM2 Hiera image encoder（待接 sam2）
}
DEFAULT_OCT_TEACHERS = ("dinov3_vitl16", "retfound_oct", "medsam2")


def build_foundmim_for_oct(student="base", teachers=DEFAULT_OCT_TEACHERS,
                           img_size=224, in_chans=3, mask_ratio=0.5,
                           loss_type="cos_sim", cross_attn_decoder_teacher=True,
                           reconstruct_orig_img=True, loss_weights=None,
                           teacher_feature_sizes=None):
    """建 OCT 域 FoundMIM 學生。回傳 (model, mask_ratio)。

    teacher_feature_sizes: 可覆寫教師 [C,H,W]（測試用小尺寸）；None → 用 OCT_TEACHERS。
    mask_ratio 預設 0.5（楊瀚博消融最佳）。
    """
    teachers = list(teachers)
    sizes_map = teacher_feature_sizes or OCT_TEACHERS
    feat_sizes = [sizes_map[t] for t in teachers]
    weights = list(loss_weights) if loss_weights else [1.0] * len(teachers)
    model = STUDENT_FACTORY[student](
        img_size=img_size, in_chans=in_chans,
        teacher_names=teachers, feature_sizes_teacher=feat_sizes,
        loss_weights_teacher=weights, loss_type_teacher=loss_type,
        cross_attn_decoder_teacher=cross_attn_decoder_teacher,
        reconstruct_orig_img=reconstruct_orig_img,
    )
    return model, mask_ratio


# 教師特徵抽取（真載入本地 ckpts）= teachers.py（RETFound/DINOv3 已驗；MedSAM2 待接 sam2）
from .teachers import TeacherFeatureExtractor   # noqa: E402,F401  (re-export)


# ───────────────────────── dummy 自測（`python -m forecast_c.phase1.distill`） ─────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    # self-test：img_size=224（學生網格 14×14，楊瀚博 translator 要求 ≥12×12）+ 小通道教師特徵圖
    test_teachers = ("dinov3_vitl16", "retfound_oct", "medsam2")
    small_sizes = {"dinov3_vitl16": [64, 14, 14], "retfound_oct": [64, 14, 14],
                   "medsam2": [32, 16, 16]}
    model, mr = build_foundmim_for_oct(student="small", teachers=test_teachers,
                                       img_size=224, mask_ratio=0.5,
                                       teacher_feature_sizes=small_sizes)
    B = 2
    imgs = torch.randn(B, 3, 224, 224)
    teacher_features = {t: torch.randn(B, h * w, c) for t, (c, h, w) in small_sizes.items()}

    loss, loss_teacher_dict, loss_balance, pred, mask = model(imgs, mask_ratio=mr,
                                                              teacher_features=teacher_features)
    assert loss is not None and torch.isfinite(loss), "MAE 重建 loss 應有限"
    assert set(loss_teacher_dict.keys()) == set(test_teachers), loss_teacher_dict.keys()
    total = loss + sum(d["patch"] for d in loss_teacher_dict.values())
    assert torch.isfinite(total)
    total.backward()
    assert model.patch_embed.proj.weight.grad is not None, "學生 ViT 應反傳"
    print("distill(FoundMIM) dummy 自測通過 ✅  教師=%s, mask=%.1f, loss=%.3f" %
          (list(test_teachers), mr, float(total)))

    # 真 extractor: MedSAM2 仍待接 sam2（RETFound/DINOv3 已可載，見 teachers.py self-test）
    try:
        TeacherFeatureExtractor(ckpt_dir="ckpts").load("medsam2")
    except (NotImplementedError, FileNotFoundError):
        print("  [OK] MedSAM2 extractor 待接 sam2；RETFound/DINOv3 已驗（teachers.py）")
