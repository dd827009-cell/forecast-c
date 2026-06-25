"""模塊 B: 厚度頭。ẑ token 網格 → 厚度圖 µm（BM−ILM 真 GT 監督）。

完整規劃_C §6: 厚度=主（真 GT）。最小版用**簡單解碼器**（O：多尺度 on/off 消融留 Phase 3）：
  token 網格 (B, grid_t·grid_h·grid_w, D) → 沿 depth(grid_t) 池化 → 2D 特徵圖 (grid_h, grid_w)
  → Conv → bilinear 上採樣到 (out_h=B-scan 數, out_w=A-scan 列) → 厚度圖 µm。

⚠️ 這是最小可跑的占位解碼器（geometry-agnostic）。Phase 3 換 **官方 SwinUNETR**（MONAI）做真正
   多尺度密集讀出 + 積水分割。token→en-face 的精確幾何對應也留 Phase 3。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from forecast_c.config import BackboneSpec, ThicknessHeadConfig


class ThicknessHead(nn.Module):
    """forward(tokens (B, N, D)) -> thickness (B, out_h, out_w)  µm。"""

    def __init__(self, backbone: BackboneSpec, cfg: ThicknessHeadConfig):
        super().__init__()
        self.gt, self.gh, self.gw = backbone.grid_t, backbone.grid_h, backbone.grid_w
        self.out_h, self.out_w = cfg.out_h, cfg.out_w
        self.proj = nn.Linear(backbone.embed_dim, cfg.hidden)
        self.to_thick = nn.Sequential(
            nn.Conv2d(cfg.hidden, cfg.hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(cfg.hidden, 1, 1))

    def forward(self, tokens):
        B, N, D = tokens.shape
        assert N == self.gt * self.gh * self.gw, (N, self.gt, self.gh, self.gw)
        h = self.proj(tokens)                                     # (B,N,hidden)
        h = h.reshape(B, self.gt, self.gh, self.gw, -1).mean(dim=1)   # depth 池化 → (B,gh,gw,hidden)
        h = h.permute(0, 3, 1, 2)                                 # (B,hidden,gh,gw)
        m = self.to_thick(h)                                     # (B,1,gh,gw)
        m = F.interpolate(m, size=(self.out_h, self.out_w),
                          mode="bilinear", align_corners=False)
        return m.squeeze(1)                                       # (B,out_h,out_w) µm


# ───────────────────────── dummy 自測（`python -m forecast_c.model.thickness`） ─────────────────────────
if __name__ == "__main__":
    from forecast_c.config import ForecastConfig
    cfg = ForecastConfig.tiny()
    head = ThicknessHead(cfg.backbone, cfg.thickness)
    B, N, D = 2, cfg.backbone.n_tokens, cfg.backbone.embed_dim
    out = head(torch.randn(B, N, D))
    assert out.shape == (B, cfg.thickness.out_h, cfg.thickness.out_w), out.shape
    # 監督可反傳（對假 GT 算 smooth_l1）
    gt = torch.rand_like(out) * 400.0
    F.smooth_l1_loss(out, gt).backward()
    assert head.proj.weight.grad is not None
    print("thickness dummy 自測通過 ✅", tuple(out.shape))
