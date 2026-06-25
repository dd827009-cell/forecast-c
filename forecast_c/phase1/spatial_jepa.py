"""空間 JEPA — 訓 Phase 1b 跨切片 Adapter3D（★ 設計 C 原創，兩篇前輩都沒有）。

完整規劃_C §4「空間 JEPA 訓 adapter」:
  遮 k 張切片 → 帶 Adapter3D 的學生**從可見鄰切片預測被遮切片的表徵**;
  target = **凍結學生（無 adapter）對被遮切片的特徵**（stop-grad，穩定靶、不崩、免 EMA）。
  只訓 adapter（+ mask token）；base 凍結（PEFT）。adapter on/off = 天然消融（K）。

為何是 JEPA 而非前輩的方法:
  - 楊瀚博 = 蒸餾+MIM（對教師/像素），吳韋論 = 分類微調。**都不是「遮切片→預測凍結特徵」的 JEPA**。
  - 這裡的「遮的是切片(跨切片結構)、預測的是凍結 base 的 latent、stop-grad target」= I-JEPA 精神搬到 OCT 深度軸。

機制（token 級，I-JEPA 式，不重建像素）:
  輸入 = 一批 volume 的 per-slice patch tokens (B_vol·d_size, N, C)（patch_embed 後）。
  target_encoder = 凍結 base blocks（無 adapter）→ 每切片 latent（detach）。
  context_encoder = 同一份凍結 base blocks + 逐 block Adapter3D（跨切片混）。
    被遮切片的輸入 token 換成 mask_token → 逼 adapter 從可見鄰切片推。
  loss = 被遮切片上 context vs target 的 cosine/smooth_l1。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter3d import Adapter3D


def slice_mask(n_vol, d_size, mask_frac=0.5, device="cpu", generator=None):
    """每 volume 隨機遮 k=clamp(round(mask_frac·d_size), 1, d_size-1) 張切片。
    回傳 (n_vol·d_size,) bool（True=被遮）。保證每 volume 至少 1 遮、1 可見（adapter 要有 context）。"""
    k = max(1, min(d_size - 1, round(mask_frac * d_size)))
    mask = torch.zeros(n_vol, d_size, dtype=torch.bool, device=device)
    for v in range(n_vol):
        idx = torch.randperm(d_size, generator=generator, device=device)[:k]
        mask[v, idx] = True
    return mask.reshape(-1)


class SpatialJEPA(nn.Module):
    """空間 JEPA 訓練 wrapper（只訓 adapter + mask_token）。

    base_blocks: 凍結 2D 學生的 transformer blocks（nn.ModuleList，介面 blk(x)->x）。
    dim, d_size: token 維度 / 每 volume 切片數。adapter_channels: bottleneck 寬。
    """

    def __init__(self, base_blocks: nn.ModuleList, dim, d_size,
                 adapter_channels=64, has_cls=True, mask_frac=0.5, loss_type="cos_sim"):
        super().__init__()
        self.blocks = base_blocks                     # 凍結 base（共用於 target/context）
        for p in self.blocks.parameters():
            p.requires_grad_(False)
        self.adapters = nn.ModuleList([
            Adapter3D(dim, adapter_channels, d_size, has_cls=has_cls)
            for _ in base_blocks])                    # 逐 block 一個 adapter（可訓）
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.d_size, self.mask_frac, self.loss_type = d_size, mask_frac, loss_type

    def _encode_target(self, tokens):
        """凍結 base（無 adapter），每切片獨立 → latent。stop-grad。"""
        with torch.no_grad():
            x = tokens
            for blk in self.blocks:
                x = blk(x)
        return x

    def _encode_context(self, tokens):
        """凍結 base + 逐 block Adapter3D（跨切片混）→ latent。adapter 有梯度。"""
        x = tokens
        for blk, ad in zip(self.blocks, self.adapters):
            x = ad(blk(x))
        return x

    def _per_token_loss(self, pred, target):
        if self.loss_type == "cos_sim":
            return 1.0 - F.cosine_similarity(pred, target, dim=-1)      # (M, N)
        return F.smooth_l1_loss(pred, target, reduction="none").mean(-1)

    def forward(self, tokens, mask=None, generator=None):
        """tokens: (B_vol·d_size, N, C) per-slice patch tokens。回傳 {loss, n_masked}。"""
        B = tokens.shape[0]
        assert B % self.d_size == 0, f"batch({B}) 須為 d_size({self.d_size}) 倍數"
        n_vol = B // self.d_size
        if mask is None:
            mask = slice_mask(n_vol, self.d_size, self.mask_frac, tokens.device, generator)

        target = self._encode_target(tokens)                            # (B,N,C) detach

        ctx_in = tokens.clone()
        ctx_in[mask] = self.mask_token.to(tokens.dtype)                 # 被遮切片 → mask_token
        pred = self._encode_context(ctx_in)                            # (B,N,C)

        per = self._per_token_loss(pred[mask], target[mask])           # 只在被遮切片
        loss = per.mean() if per.numel() > 0 else pred.sum() * 0.0
        return {"loss": loss, "n_masked": int(mask.sum())}

    def trainable_params(self):
        """只回 adapter + mask_token（base 凍結排除）。"""
        return [p for p in self.parameters() if p.requires_grad]


# ───────────────────────── dummy 自測（`python -m forecast_c.phase1.spatial_jepa`） ─────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    dim, d_size, hw = 64, 4, 4
    n_vol, N = 3, hw * hw + 1                      # +1 cls
    tokens = torch.randn(n_vol * d_size, N, dim)

    base = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
    jepa = SpatialJEPA(base, dim, d_size, adapter_channels=16, has_cls=True, mask_frac=0.5)

    # base 凍結、adapter+mask_token 可訓
    assert all(not p.requires_grad for p in jepa.blocks.parameters())
    tp = jepa.trainable_params()
    assert len(tp) > 0 and jepa.mask_token.requires_grad

    out = jepa(tokens)
    assert torch.isfinite(out["loss"]) and 0 < out["n_masked"] < n_vol * d_size
    out["loss"].backward()
    # 梯度進 adapter + mask_token，不進凍結 base
    assert all(p.grad is None for p in jepa.blocks.parameters()), "base 凍結不該有梯度"
    assert jepa.adapters[0].up.weight.grad is not None, "adapter 應反傳"
    assert jepa.mask_token.grad is not None, "mask_token 應反傳"

    # slice_mask: 每 volume 至少 1 遮 1 可見
    m = slice_mask(n_vol, d_size, 0.5).reshape(n_vol, d_size)
    assert (m.sum(1) >= 1).all() and (m.sum(1) <= d_size - 1).all()

    # 起點（adapter 零初始恆等）→ context≈target on 可見切片；loss 主要來自被遮切片（adapter 要學）
    print("spatial_jepa dummy 自測通過 ✅  (n_masked=%d/%d, loss=%.4f, ★設計C原創)" %
          (out["n_masked"], n_vol * d_size, float(out["loss"])))
