"""存亡消融對照 + baseline 群（完整規劃_C §8 / 待辦 §存亡兩層；清單見 docs/phase1_encoder_ablation.md）。

latent 預測要證明值得，必須贏這些對照。分四層：

① 地板 baseline（證「有在學東西」）:
  - **persistence / copy-last**: ẑ = z_t（不預測變化）。**P0 鐵律**: 會變的眼上贏不了 → 任務不可測。
  - **mean-change**: ẑ = z_t + μ（μ=平均 latent 變化；`fit_mean_change` 非梯度設值 / 或可學）。
  - **baseline-severity**: 只用起始 CST 線性回歸未來 CST（回歸均值）→ `BaselineSeverityRegressor`。
  - **Δt-only**: cond 只留 Δt（治療/baseline 清零）→ helper `dt_only_cond`。
  - **treatment-blind**: predictor 不看治療 → = ForecastModel(treatment=None)（eval 時切）。

③ Encoder baseline（證「自建 encoder 值得」）:
  - `build_encoder_baseline(name)`: octcube(★最關鍵)/retfound/imagenet_vit/student_full（換 encoder 餵同 predictor；L40 接點）。
  - **端到端 3D-CNN**: 不用任何 FM 直接訓 → `EndToEndVolumeRegressor`（「不如直接訓 CNN」對照）。

④ 預測方法 baseline（證「latent forecasting 值得」）:
  - **direct-regression**: 直接從 z_t(+cond) 回歸厚度，**不經 latent 預測**→ `DirectThicknessRegressor`。
    **L 鐵律**: 輸給它 → 方法無理由（重新定位賣標籤效率/多步/遷移）。

⑤ 文獻 baseline（Related Work 對比；多不可直接複現）: `LITERATURE_BASELINES` / `literature_baseline`。

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


# ═══════════════════ ① 地板 baseline（補上面 persistence / mean-change / Δt-only）═══════════════════
class BaselineSeverityRegressor(nn.Module):
    """只用 baseline 嚴重度(CST 純量)線性回歸未來 CST（回歸均值效應）。

    測『影像/治療有沒有加值』——連 baseline severity 都贏不了 → 任務退化。
    forward(baseline_cst (B,) 或 (B,1)) -> future_cst (B,)。可梯度訓，或 `fit_ols` 閉式解。
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))     # 斜率（回歸均值通常 <1）
        self.b = nn.Parameter(torch.zeros(1))    # 截距

    def forward(self, baseline_cst):
        return self.a * baseline_cst.reshape(-1) + self.b

    @torch.no_grad()
    def fit_ols(self, x, y):
        """閉式最小平方（非梯度）: x,y (M,) = baseline→future CST。"""
        x, y = x.reshape(-1).float(), y.reshape(-1).float()
        xm, ym = x.mean(), y.mean()
        a = ((x - xm) * (y - ym)).sum() / ((x - xm).pow(2).sum() + 1e-8)
        self.a.copy_(a.reshape(1)); self.b.copy_((ym - a * xm).reshape(1))
        return self


@torch.no_grad()
def fit_mean_change(mc: MeanChangeBaseline, deltas):
    """把 MeanChangeBaseline.μ 直接設成資料平均 latent 變化（非梯度）: deltas (M,D)。"""
    mc.mu.copy_(deltas.float().mean(0))
    return mc


# ═══════════════════ ③ Encoder baseline（換 encoder / 端到端）═══════════════════
# 「frozen OCTCube / RETFound-only / ImageNet-ViT → 同 predictor」= 換 encoder 餵 ForecastModel，
#   不是新類別；用 build_encoder_baseline(name) 取得該 encoder（真實權重 = Phase1/L40 接點）。
ENCODER_BASELINES = {
    "octcube":      "凍結 OCTCube（現成 3D OCT FM）→ 同 predictor。★最關鍵：student 沒打贏它就沒理由自建。",
    "retfound":     "凍結 RETFound 單教師 → 同 predictor（單教師 vs 多教師）。",
    "imagenet_vit": "ImageNet 預訓練 ViT（per-slice）→ 同 predictor（通用 encoder 對照）。",
    "student_full": "本方法：多教師蒸餾 student + 3D adapter + 空間 JEPA。",
}


def build_encoder_baseline(name, cfg):
    """回傳指定 encoder（餵 ForecastModel 當 Phase2 對照）。真實權重是 Phase1/L40 接點。"""
    raise NotImplementedError(
        f"encoder baseline '{name}': {ENCODER_BASELINES.get(name, '未知變體')}. "
        "接 forecast_c.model.encoder（OCTCube）/ phase1 產出的 student ckpt（L40）。")


class EndToEndVolumeRegressor(nn.Module):
    """不用任何 FM，從原始 volume 端到端訓小 3D-CNN → 未來厚度圖 µm。

    測『不如直接訓一個 CNN』——凍結 FM + predictor 贏不了它 → 這套沒優勢。
    forward(volume (B,1,Dp,H,W), cond (B,cond_dim)) -> thickness (B,out_h,out_w)。
    """
    def __init__(self, cfg, cond_dim, width=16):
        super().__init__()
        self.out_h, self.out_w = cfg.thickness.out_h, cfg.thickness.out_w
        C = width
        self.enc = nn.Sequential(
            nn.Conv3d(1, C, 3, 2, 1), nn.GELU(),
            nn.Conv3d(C, 2 * C, 3, 2, 1), nn.GELU(),
            nn.Conv3d(2 * C, 4 * C, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten())                    # (B,4C)
        self.head = nn.Sequential(
            nn.Linear(4 * C + cond_dim, cfg.thickness.hidden), nn.GELU(),
            nn.Linear(cfg.thickness.hidden, self.out_h * self.out_w))

    def forward(self, volume, cond):
        x = torch.cat([self.enc(volume), cond], dim=-1)
        return self.head(x).reshape(-1, self.out_h, self.out_w)       # (B,out_h,out_w) µm


# ═══════════════════ ⑤ 文獻 baseline（登錄供 Related Work 對比；多不可直接複現）═══════════════════
LITERATURE_BASELINES = {
    "anti_vegf_response_ml": "傳統 ML(RF/GBM) on 手工 OCT/臨床特徵預測 anti-VEGF 反應（多篇、特定資料集）。",
    "retfound_downstream":   "RETFound 原論文下游（疾病進展/預測）作為 FM 對照點。",
    "cnn_progression":       "CNN 端到端預測 OCT 進展 / CST（監督）。",
}


def literature_baseline(name):
    """文獻 baseline：非本 repo 可直接複現，需依原論文各自實作或僅引用對比。"""
    raise NotImplementedError(
        f"文獻 baseline '{name}': {LITERATURE_BASELINES.get(name, '未知')}. 依原論文實作或僅引用。")


# ───────────────────────── dummy 自測（`python -m forecast_c.model.baselines`） ─────────────────────────
if __name__ == "__main__":
    from forecast_c.config import ForecastConfig
    cfg = ForecastConfig.tiny()
    B, N, D = 2, cfg.backbone.n_tokens, cfg.backbone.embed_dim
    z_t = torch.randn(B, N, D)

    # persistence
    assert torch.equal(persistence(z_t), z_t)

    # mean-change: 起點 μ=0 → ẑ=z_t；fit_mean_change 設成資料平均
    mc = MeanChangeBaseline(D)
    assert torch.allclose(mc(z_t), z_t)
    mc(z_t).sum().backward(); assert mc.mu.grad is not None
    fit_mean_change(mc, torch.randn(10, D)); assert mc.mu.abs().sum() > 0

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

    # ① baseline-severity 回歸 + 閉式 fit（斜率應≈0.6）
    bs = BaselineSeverityRegressor()
    xx = torch.linspace(200, 600, 32); yy = 0.6 * xx + 120 + torch.randn(32)
    bs.fit_ols(xx, yy)
    assert abs(bs.a.item() - 0.6) < 0.15 and bs(xx).shape == (32,)

    # ③ 端到端 3D-CNN baseline: 任意 volume → 厚度圖 + 反傳
    e2e = EndToEndVolumeRegressor(cfg, cfg.predictor.cond_dim)
    o2 = e2e(torch.randn(B, 1, 16, 64, 64), cond)
    assert o2.shape == (B, cfg.thickness.out_h, cfg.thickness.out_w)
    o2.sum().backward(); assert e2e.enc[0].weight.grad is not None

    # ③⑤ 換 encoder / 文獻 baseline = 清楚報 NotImplemented
    try: build_encoder_baseline("octcube", cfg); assert False
    except NotImplementedError: pass
    try: literature_baseline("retfound_downstream"); assert False
    except NotImplementedError: pass

    print("baselines dummy 自測通過 ✅  (floor + encoder + e2e + literature)")
