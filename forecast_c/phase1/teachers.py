"""教師特徵抽取（從本地 ckpts 載入，永不連 HF）。

楊瀚博式:教師凍結 → 對每張 B-scan 抽 patch 特徵 → [B, H*W, C]（蒸餾目標）。
本檔接**本地權重**（`ckpts/`），離線可跑:
  - RETFound (OCT)  : 標準 MAE ViT-L/16 → timm `vit_large_patch16` 直接載。 ✅ CPU
  - DINOv3 ViT-L/16 : 官方 checkpoint → timm `vit_large_patch16_dinov3` + 鍵 remap。 ✅ CPU
  - MedSAM2         : SAM2 Hiera image encoder → 需 `sam2` 套件（較重）。 ⏳ documented

B-scan(灰階) → 複製 3ch → resize img_size → ImageNet 正規化（三者皆 RGB 預訓）。
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ckpt 檔名（放 ckpts/）
CKPT_FILES = {
    "retfound_oct":  "RETFound_mae_natureOCT.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "medsam2":       "MedSAM2_latest.pt",
}
# timm 架構名（ViT 教師）
TIMM_ARCH = {
    "retfound_oct":  "vit_large_patch16_224",
    "dinov3_vitl16": "vit_large_patch16_dinov3",
}
# MedSAM2 = SAM2 Hiera-tiny（embed96/12blocks）image encoder。官方 sam2 config + 本地 MedSAM2 權重。
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_INPUT = 1024                       # SAM2 原生輸入 → vision_features 64×64×256


def _unwrap(sd):
    """{model:...}/{state_dict:...} → 內層 state_dict。"""
    if isinstance(sd, dict):
        for w in ("model", "state_dict", "teacher", "model_state_dict"):
            if w in sd and isinstance(sd[w], dict):
                return sd[w]
    return sd


class TeacherFeatureExtractor(nn.Module):
    """forward/extract: B-scan → 各教師 patch 特徵 [B, H*W, C]。

    ckpt_dir: 本地權重目錄（預設 'ckpts'）。lazy load（用到才載）。
    """

    def __init__(self, ckpt_dir="ckpts", img_size=224, device="cpu"):
        super().__init__()
        self.ckpt_dir, self.img_size, self.device = ckpt_dir, img_size, device
        self._models = {}                                     # name -> nn.Module（凍結）
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    # ---- 載入（本地檔，凍結）----
    def load(self, name):
        if name in self._models:
            return self._models[name]
        path = os.path.join(self.ckpt_dir, CKPT_FILES[name])
        if not os.path.exists(path):
            raise FileNotFoundError(f"教師權重不在: {path}")
        if name == "medsam2":
            # SAM2 Hiera image encoder（需 sam2 + hydra-core + omegaconf + iopath）
            try:
                from sam2.build_sam import build_sam2
            except ImportError as e:
                raise NotImplementedError(
                    "MedSAM2 需 `pip install --no-deps sam2 && pip install hydra-core omegaconf iopath`。") from e
            m = build_sam2(SAM2_CFG, path, device=self.device).image_encoder
        else:
            import timm
            m = timm.create_model(TIMM_ARCH[name], pretrained=False, num_classes=0)
            sd = _unwrap(torch.load(path, map_location="cpu", weights_only=False))
            if name == "dinov3_vitl16":
                sd = _dinov3_to_timm(sd, m)
            msg = m.load_state_dict(sd, strict=False)
            # 只在意「實質權重」缺失（head/fc_norm/rope buffer 不算；rope timm 自算）
            miss = [k for k in msg.missing_keys
                    if not k.startswith(("head", "fc_norm")) and "rope" not in k]
            assert len(miss) == 0, f"{name} 載入缺實質鍵: {miss[:10]}"
        m.eval().to(self.device)
        for p in m.parameters():
            p.requires_grad_(False)
        self._models[name] = m
        return m

    def _preprocess(self, imgs, size=None):
        """imgs: (B,1或3,H,W) 任意尺度 → (B,3,size,size) ImageNet 正規化（size 預設 self.img_size）。"""
        size = size or self.img_size
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(1)                          # (B,H,W)→(B,1,H,W)
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)                    # 灰階→3ch
        # 逐影像 min-max 到 [0,1]（B-scan 強度尺度不一）
        b = imgs.shape[0]
        flat = imgs.reshape(b, -1)
        lo, hi = flat.min(1)[0].view(b, 1, 1, 1), flat.max(1)[0].view(b, 1, 1, 1)
        imgs = (imgs - lo) / (hi - lo + 1e-6)
        imgs = F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)
        return (imgs - self.mean) / self.std

    @torch.no_grad()
    def extract(self, name, imgs):
        """imgs (B,1/3,H,W) → patch 特徵 [B, H*W, C]（對齊 OCT_TEACHERS）。"""
        m = self.load(name)
        imgs = imgs.to(self.device)
        if name == "medsam2":
            # SAM2 image encoder: 1024 輸入 → vision_features (B,256,64,64) → (B,4096,256)
            vf = m(self._preprocess(imgs, size=SAM2_INPUT))["vision_features"]
            return vf.flatten(2).transpose(1, 2)             # (B, 64*64, 256)
        feat = m.forward_features(self._preprocess(imgs))    # ViT: (B, prefix+L, C)
        n_prefix = getattr(m, "num_prefix_tokens", 1)        # cls(+registers)
        return feat[:, n_prefix:]                            # (B, L, C)


def _dinov3_to_timm(sd, model):
    """官方 DINOv3 state_dict 鍵 → timm 鍵。

    官方 → timm 對應（實測）:
      storage_tokens → reg_token；blocks.*.ls1.gamma → gamma_1；ls2.gamma → gamma_2。
    丟棄 timm 沒有的鍵: attn.qkv.bias_mask（dinov3 的 key-bias 遮罩）、rope_embed.periods（timm 自算 RoPE）、
      mask_token（推論不用）。
    """
    out = {}
    for k, v in sd.items():
        if k.endswith("attn.qkv.bias_mask") or k == "rope_embed.periods" or k == "mask_token":
            continue
        k = k.replace("storage_tokens", "reg_token")
        k = k.replace(".ls1.gamma", ".gamma_1").replace(".ls2.gamma", ".gamma_2")
        out[k] = v
    return out


# ───────────────────────── self-test（容器內，需 ckpts/）`python -m forecast_c.phase1.teachers` ─────────────────────────
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ext = TeacherFeatureExtractor(ckpt_dir="ckpts", img_size=224)
    B = 2
    bscan = torch.rand(B, 1, 496, 512)                        # 假 B-scan（驗 shape/前處理）
    for name, exp_c in [("retfound_oct", 1024), ("dinov3_vitl16", 1024)]:
        feat = ext.extract(name, bscan)
        L, C = feat.shape[1], feat.shape[2]
        hw = int(L ** 0.5)
        print(f"  [OK] {name}: feat={tuple(feat.shape)} → {hw}x{hw}x{C}")
        assert C == exp_c and hw * hw == L
    # MedSAM2（需 sam2；沒裝就跳過）
    try:
        feat = ext.extract("medsam2", bscan)
        print(f"  [OK] medsam2: feat={tuple(feat.shape)} → 64x64x256")
        assert feat.shape == (B, 64 * 64, 256)
        print("teachers self-test 通過 ✅（三教師 DINOv3+RETFound+MedSAM2 本地載入）")
    except NotImplementedError:
        print("  [skip] medsam2 需 sam2（pip install sam2 hydra-core omegaconf iopath）")
        print("teachers self-test 通過 ✅（RETFound + DINOv3；MedSAM2 待 sam2）")
