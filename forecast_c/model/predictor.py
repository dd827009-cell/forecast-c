"""模塊 D: 一步治療條件 predictor。z_t + cond → ẑ_{t+Δ}（只預測 256 語意分支 latent）。

完整規劃_C §5:
  - 淺而窄 transformer（比 encoder 窄，不搶 backbone 風頭）。
  - 單尺度治療條件注入（AdaLN-Zero / FiLM）。
  - 殘差零初始（out_proj 零初始 → Δ=0 起點 ẑ=z_t = persistence trivial 起點，逼學變化）。
  - 可選 per-token logvar 頭（→ losses 走 Gaussian NLL）。
  - 一步（多步 rollout 待 A-1 確認 ≥3 visit 才加）。
"""
import torch
import torch.nn as nn

from forecast_c.config import PredictorConfig


class AdaLNZeroBlock(nn.Module):
    """transformer block，用條件向量做 AdaLN-Zero 調變（DiT 風格）。
    gate 由條件投影且零初始 → 起點整個 block = 恆等（不破壞 backbone z_t）。"""

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        W = cfg.width
        self.norm1 = nn.LayerNorm(W, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(W, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(W, cfg.num_heads, batch_first=True)
        hidden = int(W * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(W, hidden), nn.GELU(), nn.Linear(hidden, W))
        # 一次產 6 組調變（attn/mlp 各 shift,scale,gate）；零初始 → 起點恆等
        self.adaln = nn.Linear(cfg.cond_dim, 6 * W)
        nn.init.zeros_(self.adaln.weight); nn.init.zeros_(self.adaln.bias)

    def forward(self, x, cond):
        sh1, sc1, g1, sh2, sc2, g2 = self.adaln(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc1.unsqueeze(1)) + sh1.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * attn_out
        h = self.norm2(x) * (1 + sc2.unsqueeze(1)) + sh2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


class FiLMZeroBlock(nn.Module):
    """FiLM 條件 block: 對 sublayer 輸出做 (1+γ)·out+β，gate 零初始 → 起點恆等（ablation）。"""

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        W = cfg.width
        self.norm1 = nn.LayerNorm(W)
        self.norm2 = nn.LayerNorm(W)
        self.attn = nn.MultiheadAttention(W, cfg.num_heads, batch_first=True)
        hidden = int(W * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(W, hidden), nn.GELU(), nn.Linear(hidden, W))
        self.film = nn.Linear(cfg.cond_dim, 6 * W)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        g1, b1, ga1, g2, b2, ga2 = self.film(cond).chunk(6, dim=-1)
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + ga1.unsqueeze(1) * ((1 + g1.unsqueeze(1)) * a + b1.unsqueeze(1))
        m = self.mlp(self.norm2(x))
        x = x + ga2.unsqueeze(1) * ((1 + g2.unsqueeze(1)) * m + b2.unsqueeze(1))
        return x


def make_cond_block(cfg: PredictorConfig):
    """依 cfg.cond_mode 回傳條件注入 block（adaln_zero / film）。"""
    return FiLMZeroBlock(cfg) if getattr(cfg, "cond_mode", "adaln_zero") == "film" else AdaLNZeroBlock(cfg)


class OneStepPredictor(nn.Module):
    """一步 predictor。forward(z_t, cond) -> (z_hat (B,N,in_dim), logvar (B,N,1)|None)。

    z_t  : (B, N, in_dim)   現在的 token 網格（凍結 encoder 輸出，已 LN，見 forecast_model）
    cond : (B, cond_dim)    [治療 a, Fourier(Δt), baseline_severity]
    註: z_t 已含 encoder 位置資訊，predictor 不另加 pos_embed。
    """

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg
        self.residual = getattr(cfg, "residual", True)
        self.predict_variance = getattr(cfg, "predict_variance", False)
        self.in_proj = nn.Linear(cfg.in_dim, cfg.width)
        self.blocks = nn.ModuleList([make_cond_block(cfg) for _ in range(cfg.depth)])
        self.out_norm = nn.LayerNorm(cfg.width)
        self.out_proj = nn.Linear(cfg.width, cfg.in_dim)
        if self.residual:                                  # Δ=0 起點 → ẑ=z_t（persistence trivial 起點）
            nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)
        if self.predict_variance:                          # per-token logvar 頭
            self.logvar_proj = nn.Linear(cfg.width, 1)
            nn.init.zeros_(self.logvar_proj.weight); nn.init.zeros_(self.logvar_proj.bias)  # logvar=0 → σ²=1 起點

    def forward(self, z_t, cond):
        x = self.in_proj(z_t)
        for blk in self.blocks:
            x = blk(x, cond)
        h = self.out_norm(x)
        delta = self.out_proj(h)
        z_hat = z_t + delta if self.residual else delta
        logvar = self.logvar_proj(h) if self.predict_variance else None
        return z_hat, logvar


# ───────────────────────── dummy 自測（`python -m forecast_c.model.predictor`） ─────────────────────────
if __name__ == "__main__":
    from forecast_c.config import ForecastConfig
    cfg = ForecastConfig.tiny()
    B, N = 2, cfg.backbone.n_tokens

    # AdaLN-Zero 起點恆等
    blk = AdaLNZeroBlock(cfg.predictor)
    x = torch.randn(B, N, cfg.predictor.width)
    cond = torch.randn(B, cfg.predictor.cond_dim)
    assert torch.allclose(blk(x, cond), x, atol=1e-5), "AdaLN gate 初始非0"

    # FiLM 起點恆等
    fblk = FiLMZeroBlock(cfg.predictor)
    assert torch.allclose(fblk(x, cond), x, atol=1e-5), "FiLM gate 初始非0"

    # predictor: 殘差零初始 → 起點 ẑ=z_t + logvar 形狀
    pred = OneStepPredictor(cfg.predictor)
    z_t = torch.randn(B, N, cfg.predictor.in_dim)
    z_hat, logvar = pred(z_t, cond)
    assert z_hat.shape == z_t.shape and logvar.shape == (B, N, 1)
    assert torch.allclose(z_hat, z_t, atol=1e-6), "殘差起點應 = z_t"
    z_hat.sum().backward()

    # film 模式可跑
    cfg2 = ForecastConfig.tiny(); cfg2.predictor.cond_mode = "film"
    p2 = OneStepPredictor(cfg2.predictor)
    zh, _ = p2(torch.randn(B, N, cfg2.predictor.in_dim), torch.randn(B, cfg2.predictor.cond_dim))
    assert zh.shape == (B, N, cfg2.predictor.in_dim)
    print("predictor dummy 自測通過 ✅")
