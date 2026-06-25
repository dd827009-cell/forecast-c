"""Phase 2 整機: 凍結 encoder + 一步 predictor + 厚度頭。

完整規劃_C §5 的關鍵設計（與前幾代 latent_dynamics 的差異）:
  ★ **免 EMA**: target = **同一個凍結 encoder** 編實際未來 + detach + LayerNorm（stop-grad 自動成立）。
     沒有 EMA teacher、沒有 VICReg。防崩塌靠 stop-grad（+ 之後 Phase1 接地）。
  - z_t 與 target **同在 LayerNorm 空間** → 未變的眼 predict loss≈0（persistence trivial 起點）。
  - 殘差零初始 predictor（起點 ẑ=z_t）+ change-weighted loss（打 persistence）。
  - 一步預測（多步 rollout 待 A-1 確認 ≥3 visit 才加）。

對接:
  §7.2 治療可關: forward(treatment=None) → naive embedding（對照組）。
  §7.3 ẑ→厚度: 注入 thickness_head，decode_thickness(ẑ) 供事實預測準度。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from forecast_c.config import ForecastConfig
from .encoder import FrozenEncoder
from .treatment import TreatmentEncoder
from .predictor import OneStepPredictor
from . import losses


def fourier_features(x, bands):
    """純量 → Fourier 特徵 (B, 2*bands)。Δt 不直接餵原始值（範圍大難學）。x: (B,) 或 (B,1)。"""
    x = x.reshape(-1, 1)
    freqs = (2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)) * math.pi
    ang = x * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class ForecastModel(nn.Module):
    def __init__(self, encoder: nn.Module, cfg: ForecastConfig, thickness_head: nn.Module = None):
        super().__init__()
        self.cfg = cfg
        # encoder 必須凍結；非 FrozenEncoder 則包一層（冪等：FrozenEncoder 再包仍凍結）
        self.encoder = encoder if isinstance(encoder, FrozenEncoder) else FrozenEncoder(encoder)
        self.treat = TreatmentEncoder(cfg.treat)
        self.predictor = OneStepPredictor(cfg.predictor)
        self.thickness_head = thickness_head

    # ---- 條件向量 = [治療 a, Fourier(Δt), baseline] ----
    def build_cond(self, a, dt, baseline):
        # dt 須已正規化（年；dt_unit="years"）才餵 Fourier；餵原始天數會高頻 aliasing。lenient 守門。
        assert torch.isfinite(dt).all() and float(dt.abs().max()) < 100.0, \
            "dt 應為正規化(年)再餵 Fourier；看起來像原始天數 → 會 aliasing"
        dt_feat = fourier_features(dt, self.cfg.predictor.dt_fourier_bands)
        baseline = baseline.reshape(baseline.shape[0], -1)        # (B,)/(B,k) → (B, baseline_dim)
        return torch.cat([a, dt_feat, baseline], dim=-1)

    def _treatment_vec(self, treatment, B):
        if treatment is None:                                     # §7.2 治療可關 → naive 對照
            assert self.cfg.treat.use_naive_embedding, "治療可關需 use_naive_embedding"
            return self.treat.naive_emb.unsqueeze(0).expand(B, -1)
        return self.treat(treatment)

    @torch.no_grad()
    def encode_target(self, volume_future):
        """★ stop-grad target: 同一凍結 encoder 編未來 + LayerNorm（免 EMA）。"""
        tokens, _ = self.encoder(volume_future)                   # 已 no_grad（凍結）
        return F.layer_norm(tokens, (tokens.shape[-1],))

    def encode_present(self, volume_t):
        """現在 latent z_t（LN 空間，與 target 一致）。"""
        tokens, _ = self.encoder(volume_t)
        return F.layer_norm(tokens, (tokens.shape[-1],))

    def forward(self, volume_t, volume_future, treatment, dt, baseline,
                target_mask=None, thickness_gt=None, thickness_mask=None):
        z_t = self.encode_present(volume_t)                       # (B,N,D) LN 空間
        B = z_t.shape[0]
        a = self._treatment_vec(treatment, B)
        cond = self.build_cond(a, dt, baseline)
        z_hat, logvar = self.predictor(z_t, cond)                 # 殘差 ẑ=z_t+Δ；logvar 或 None
        z_target = self.encode_target(volume_future)              # stop-grad + LN

        predict = losses.predict_loss(z_hat, z_target, z_t, logvar, target_mask, self.cfg.loss)
        total = self.cfg.loss.w_predict * predict
        detail = {"predict": float(predict.detach())}

        # §7.3 厚度頭監督（有 GT 才算；ẑ→厚度 µm）
        if self.thickness_head is not None and thickness_gt is not None:
            thick = self.decode_thickness(z_hat)                  # (B,out_h,out_w)
            per = F.smooth_l1_loss(thick, thickness_gt, reduction="none")
            if thickness_mask is not None:
                mf = thickness_mask.float()
                thick_loss = (per * mf).sum() / mf.sum().clamp_min(1.0)
            else:
                thick_loss = per.mean()
            total = total + self.cfg.loss.w_thickness * thick_loss
            detail["thickness"] = float(thick_loss.detach())

        return {"z_hat": z_hat, "logvar": logvar, "z_target": z_target,
                "loss": total, "detail": detail}

    def decode_thickness(self, z_hat):
        """§7.3 事實預測準度: ẑ → 厚度圖 µm（與真實未來比）。"""
        assert self.thickness_head is not None, "需注入 thickness_head"
        return self.thickness_head(z_hat)


# ───────────────────────── dummy 自測（`python -m forecast_c.model.forecast_model`） ─────────────────────────
if __name__ == "__main__":
    from forecast_c.config import ForecastConfig
    from .encoder import DummyEncoder
    from .thickness import ThicknessHead

    cfg = ForecastConfig.tiny()
    N, D = cfg.backbone.n_tokens, cfg.backbone.embed_dim
    th = ThicknessHead(cfg.backbone, cfg.thickness)
    model = ForecastModel(DummyEncoder(D), cfg, thickness_head=th)

    # encoder 凍結確認
    assert all(not p.requires_grad for p in model.encoder.parameters())

    B = 2
    vt, vf = torch.randn(B, N, D), torch.randn(B, N, D)
    treat = {"drug_ids": torch.randint(1, cfg.treat.n_drug_types, (B, 3)),
             "numerics": torch.rand(B, 3, cfg.treat.numeric_in),
             "event_mask": torch.ones(B, 3, dtype=torch.bool),
             "is_naive": torch.zeros(B, dtype=torch.bool)}
    dt = torch.rand(B); base = torch.rand(B)
    gt = torch.rand(B, cfg.thickness.out_h, cfg.thickness.out_w) * 400.0

    out = model(vt, vf, treat, dt, base, thickness_gt=gt)
    assert out["z_hat"].shape == (B, N, D)
    assert "thickness" in out["detail"]
    out["loss"].backward()
    # 梯度進 predictor/treat/thickness，不進凍結 encoder
    assert all(p.grad is None for p in model.encoder.parameters()), "凍結 encoder 不該有梯度"
    assert model.predictor.out_proj.weight.grad is not None
    assert model.thickness_head.proj.weight.grad is not None

    # §7.2 治療可關
    out2 = model(vt, vf, None, dt, base)
    assert out2["z_hat"].shape == (B, N, D)

    # persistence 起點: future==present → predict 損失≈0（LN 空間殘差零初始）
    model.zero_grad()
    v = torch.randn(B, N, D)
    out3 = model(v, v, None, dt, base)
    assert out3["detail"]["predict"] < 1e-4, out3["detail"]["predict"]

    # ★ 免 EMA: 沒有 teacher 模組
    assert not hasattr(model, "teacher"), "最小版不該有 EMA teacher"
    print("forecast_model dummy 自測通過 ✅  (★免EMA, persistence起點%.2e)" % out3["detail"]["predict"])
