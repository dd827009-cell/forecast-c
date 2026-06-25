"""存亡消融對照（完整規劃_C §8 / 待辦 §存亡兩層）。

latent 預測要證明值得，必須贏這些對照:
  - **persistence / copy-last**: ẑ = z_t（不預測變化）。**P0 鐵律**: 會變的眼上贏不了 → 任務不可測。
  - **mean-change**: ẑ = z_t + μ（μ = 訓練集平均 latent 變化）。比 persistence 強一點的 trivial。
  - **treatment-blind**: predictor 不看治療 → = ForecastModel(treatment=None)（eval 時切，不需另類）。
  - **Δt-only**: cond 只留 Δt（治療/baseline 清零）→ helper `dt_only_cond`。
  - **direct-regression**: 直接從 z_t(+cond) 回歸厚度變化，**不經 latent 預測**。**L 鐵律**: 輸給它 →
    方法無理由（重新定位賣標籤效率/多步/遷移）。

這些對照與 ForecastModel 共用同一套評估（見 train/eval.py）。
"""
import torch
import torch.nn as nn

from forecast_c.config import BackboneSpec, ThicknessHeadConfig


def persistence(z_t):
    """copy-last: ẑ = z_t。無參數。"""
    return z_t


class MeanChangeBaseline(nn.Module):
    """ẑ = z_t + μ，μ (D,) 為可學平均變化（用 predict loss 訓 → 收斂到 E[z_target−z_t]）。"""

    def __init__(self, in_dim: int):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(in_dim))

    def forward(self, z_t):
        return z_t + self.mu


class DirectThicknessRegressor(nn.Module):
    """L 對照: 直接從 z_t(pooled) + cond 回歸**未來厚度圖 µm**，不經 latent 預測。

    forward(z_t (B,N,D), cond (B,cond_dim)) -> thickness (B, out_h, out_w)。
    """

    def __init__(self, backbone: BackboneSpec, cfg: ThicknessHeadConfig, cond_dim: int):
        super().__init__()
        self.out_h, self.out_w = cfg.out_h, cfg.out_w
        self.net = nn.Sequential(
            nn.Linear(backbone.embed_dim + cond_dim, cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, cfg.out_h * cfg.out_w))

    def forward(self, z_t, cond):
        pooled = z_t.mean(dim=1)                                  # (B,D) 全局 pool
        x = torch.cat([pooled, cond], dim=-1)
        return self.net(x).reshape(-1, self.out_h, self.out_w)    # (B,out_h,out_w) µm


def dt_only_cond(cond, treat_dim, dt_dim):
    """Δt-only 對照: 治療 a（前 treat_dim）與 baseline（dt 之後）清零，只留 Fourier(Δt)。"""
    c = cond.clone()
    c[:, :treat_dim] = 0.0                                        # 清治療 a
    c[:, treat_dim + dt_dim:] = 0.0                               # 清 baseline
    return c


# ───────────────────────── dummy 自測（`python -m forecast_c.model.baselines`） ─────────────────────────
if __name__ == "__main__":
    from forecast_c.config import ForecastConfig
    cfg = ForecastConfig.tiny()
    B, N, D = 2, cfg.backbone.n_tokens, cfg.backbone.embed_dim
    z_t = torch.randn(B, N, D)

    # persistence
    assert torch.equal(persistence(z_t), z_t)

    # mean-change: 起點 μ=0 → ẑ=z_t；訓練後 μ≠0
    mc = MeanChangeBaseline(D)
    assert torch.allclose(mc(z_t), z_t)
    mc(z_t).sum().backward(); assert mc.mu.grad is not None

    # direct regressor: 厚度圖形狀 + 反傳
    dr = DirectThicknessRegressor(cfg.backbone, cfg.thickness, cfg.predictor.cond_dim)
    cond = torch.randn(B, cfg.predictor.cond_dim)
    out = dr(z_t, cond)
    assert out.shape == (B, cfg.thickness.out_h, cfg.thickness.out_w)
    out.sum().backward(); assert dr.net[0].weight.grad is not None

    # Δt-only cond: 治療/baseline 清零
    c2 = dt_only_cond(cond, cfg.treat.out_dim, cfg.predictor.dt_dim)
    assert torch.all(c2[:, :cfg.treat.out_dim] == 0)
    assert torch.all(c2[:, cfg.treat.out_dim + cfg.predictor.dt_dim:] == 0)
    print("baselines dummy 自測通過 ✅")
