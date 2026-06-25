"""L_predict（latent 預測核心）。完整規劃_C §5 / §8。

- change-weighted: per-token 依 ‖z_target − z_t‖（latent 變化量）加權 → 「會變的 token」主導，
  打 persistence（否則不變的多數把損失稀釋成「複製現在」）。
- Gaussian NLL: predictor 給 per-token logvar 時走異方差 NLL（緩解確定性回歸模糊平均）；
  沒給（logvar=None）則回退 smooth_l1 / cosine。
- 防崩塌靠 stop-grad target（+ Phase1 接地），**無 VICReg、無 EMA**。
"""
import torch
import torch.nn.functional as F

from forecast_c.config import LossConfig


def predict_loss(z_hat, z_target, z_t=None, logvar=None, target_mask=None,
                 cfg: LossConfig = None):
    """latent 空間預測損失（per-token → 加權/遮罩平均）。

    z_hat    : (B,N,D) predictor 輸出 ẑ_{t+Δ}
    z_target : (B,N,D) 凍結 encoder 編的未來 latent（呼叫端已 stop-grad + LayerNorm）
    z_t      : (B,N,D) 現在 latent（change-weighting 的基準）；None → 不加權
    logvar   : (B,N,1) per-token log σ²；None → 走 predict_kind
    target_mask: (B,N) bool 或 None
    """
    cfg = cfg or LossConfig()
    if logvar is not None:
        # 異方差 Gaussian NLL（per-token 共享 σ²，跨 dim 平均平方誤差）
        s = logvar.squeeze(-1)                                   # (B,N)
        se = ((z_hat - z_target) ** 2).mean(dim=-1)             # (B,N)
        per = 0.5 * (torch.exp(-s) * se + s)
    elif cfg.predict_kind == "cosine":
        per = 1.0 - F.cosine_similarity(z_hat, z_target, dim=-1)
    else:
        per = F.smooth_l1_loss(z_hat, z_target, reduction="none").mean(dim=-1)

    # change-weighting: 變化大的 token 加權（權重 detach，僅重分配不改總尺度）
    if cfg.change_weighted and z_t is not None:
        with torch.no_grad():
            chg = (z_target - z_t).norm(dim=-1)                 # (B,N)
            w = chg / (chg.mean(dim=1, keepdim=True) + 1e-6)    # 每-sample 正規化 → 均值≈1
            # 防呆: clamp 到 [min,max] → 不變的眼有 floor（穩定區梯度）、避免少數 token 爆尖峰
            w = w.clamp(min=cfg.change_w_min, max=cfg.change_w_max)
        per = per * w

    if target_mask is not None:
        mf = target_mask.float()
        return (per * mf).sum() / mf.sum().clamp_min(1.0)
    return per.mean()


# ───────────────────────── dummy 自測（`python -m forecast_c.model.losses`） ─────────────────────────
if __name__ == "__main__":
    cfg = LossConfig()
    z = torch.randn(2, 64, 32)
    assert float(predict_loss(z, z.clone(), cfg=cfg)) < 1e-6, "相同應≈0"
    assert float(predict_loss(torch.randn_like(z), z, cfg=cfg)) > 0

    # Gaussian NLL: 誤差大時調高 σ²（logvar 大）應降 NLL
    z_hat = torch.randn(2, 16, 32); z_tgt = z_hat + 1.0
    lo = predict_loss(z_hat, z_tgt, logvar=torch.full((2, 16, 1), -2.0), cfg=cfg)
    hi = predict_loss(z_hat, z_tgt, logvar=torch.full((2, 16, 1), 2.0), cfg=cfg)
    assert float(hi) < float(lo), "誤差大時調高 σ² 應降 NLL"

    # change-weighted: 變化集中在 token0 時，change-weight 放大其損失
    z_t = torch.zeros(1, 4, 8); z_tgt2 = z_t.clone(); z_tgt2[0, 0] = 10.0
    z_hat2 = z_t.clone()
    on = float(predict_loss(z_hat2, z_tgt2, z_t=z_t, cfg=cfg))
    cfg_off = LossConfig(change_weighted=False)
    off = float(predict_loss(z_hat2, z_tgt2, z_t=z_t, cfg=cfg_off))
    assert on > off, "change-weight 應放大會變 token 的損失"

    # floor: 完全不變的眼仍有限非零（不 NaN/不歸零）
    z0 = torch.zeros(1, 8, 4)
    l = float(predict_loss(torch.randn(1, 8, 4), z0.clone(), z_t=z0, cfg=cfg))
    assert l == l and 0.0 < l < 1e6
    print("losses dummy 自測通過 ✅")
