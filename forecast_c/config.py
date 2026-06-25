"""forecast_c 最小版超參（單一真相）。對應設計 C `latent_forecast_C/完整規劃_C.md` §5。

最小版 = 凍結 OCTCube → 一步治療條件 predictor（殘差 + change-weighted）→ 厚度頭。
與前幾代（latent_dynamics）的關鍵差異:
  - ★ 免 EMA: target 由「同一個凍結 encoder」編未來 + stop-grad，故**沒有 TeacherConfig**。
  - 無 VICReg。
  - 不含多步 rollout / 軌跡 SSM / 存活頭 / 雙頭空間輔助（待 A-1 普查決策後再加）。

λ 權重 / Δt curriculum 等數值待 L40 + A-1 實測定。
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BackboneSpec:
    """凍結 student/encoder（最小版 = 官方 OCTCube ViT-L）的唯讀對接資訊，非可調。"""
    embed_dim: int = 1024            # ViT-L token 維度
    # 256 語意分支 token 網格: input256/patch16=16x16 空間 × (60/3=20) 時間
    grid_t: int = 20
    grid_h: int = 16
    grid_w: int = 16
    cls_embed: bool = True           # encoder 額外輸出 CLS（z_t 用 token 網格，不含 CLS）

    @property
    def n_tokens(self) -> int:       # 不含 CLS
        return self.grid_t * self.grid_h * self.grid_w   # 5120


@dataclass
class TreatmentEncoderConfig:
    """模塊 C。藥身份 與 次數/天數 分開編碼（反事實鋪路）。

    資料保證知道「用哪一款藥」→ 藥身份是主訊號；次數/天數由回診紀錄推導；劑量不做。
    """
    n_drug_types: int = 16           # 藥物種類數（含 padding=0）; 待真實藥表定
    drug_embed_dim: int = 32         # 藥身份 embedding 維度（主訊號）
    numeric_in: int = 3              # [打針次數, 距最後一針天數, 累積針數]（無劑量）
    numeric_hidden: int = 32
    out_dim: int = 64                # 治療向量 a 維度（餵條件注入）
    use_naive_embedding: bool = True # treatment-naive 專屬「無治療」embedding（對照組 / 治療可關）
    aggregate: str = "time_weighted_sum"  # 多藥聚合（越近越重）


@dataclass
class PredictorConfig:
    """模塊 D。淺而窄的 transformer + 條件注入。只預測 256 分支 token 網格。一步。"""
    width: int = 512                 # 比 encoder(1024) 窄（不搶 backbone 風頭）
    depth: int = 8                   # 6~12 之間
    num_heads: int = 8
    mlp_ratio: float = 4.0
    # 殘差預測: ẑ = z_t + Δ（out_proj 零初始 → Δ=0 起點 → ẑ=z_t = persistence trivial 起點，逼學變化）
    residual: bool = True
    # 機率性預測: per-token logvar 頭，loss 走 Gaussian NLL（緩解確定性回歸模糊平均）。
    predict_variance: bool = True
    cond_mode: str = "adaln_zero"    # 條件注入: "adaln_zero" | "film"（ablation）
    # Δt（天）不直接餵原始數字，先 Fourier 編碼（範圍大，原始 scalar 難學）
    dt_fourier_bands: int = 6        # → dt_dim = 2*bands = 12
    # 基線嚴重度 = 中央亞區厚度 CST（1 純量；可擴成 CST+弱病理小向量 → 調大 baseline_dim 去混淆）
    baseline_dim: int = 1
    # 下面 3 個由 ForecastConfig.__post_init__ 自動推導，勿手填
    in_dim: int = 0                  # = backbone.embed_dim
    dt_dim: int = 0                  # = 2 * dt_fourier_bands
    cond_dim: int = 0                # = treat.out_dim + dt_dim + baseline_dim


@dataclass
class ThicknessHeadConfig:
    """模塊 B（厚度頭）。ẑ token 網格 → 厚度圖 µm（BM−ILM 真 GT 監督）。"""
    out_h: int = 25                  # 輸出厚度圖高（B-scan 數）
    out_w: int = 512                 # 輸出厚度圖寬（A-scan/列）
    hidden: int = 256


@dataclass
class LossConfig:
    predict_kind: str = "smooth_l1"  # logvar=None 時回退: "smooth_l1" / "cosine"; predict_variance 時走 Gaussian NLL
    w_predict: float = 1.0
    w_thickness: float = 1.0         # 厚度頭監督權重（事實預測準度；有 GT 才開）
    # change-weighted: 損失按 ‖z_target−z_t‖ 對「會變的 token」加權（打 persistence）。
    change_weighted: bool = True
    # 數值防呆: 權重 clamp 到 [min,max] → 不變的眼仍有 floor 梯度、避免少數 token 爆尖峰。
    change_w_min: float = 0.1
    change_w_max: float = 5.0


@dataclass
class ForecastConfig:
    backbone: BackboneSpec = field(default_factory=BackboneSpec)
    treat: TreatmentEncoderConfig = field(default_factory=TreatmentEncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    thickness: ThicknessHeadConfig = field(default_factory=ThicknessHeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    # Δt 正規化單位（餵 Fourier 前）: "years"（預設，回診間隔遠 < 100）。
    dt_unit: str = "years"

    def __post_init__(self):
        # 衍生值從來源自動推導（改一處全跟著動，不會手寫死數字對不上）
        self.predictor.in_dim = self.backbone.embed_dim
        self.predictor.dt_dim = 2 * self.predictor.dt_fourier_bands
        self.predictor.cond_dim = (self.treat.out_dim
                                   + self.predictor.dt_dim
                                   + self.predictor.baseline_dim)

    @classmethod
    def tiny(cls):
        """測試用迷你設定 — 小 token 網格 + 窄淺 predictor，dummy 單元測快又對齊。"""
        c = cls()
        c.backbone.grid_t, c.backbone.grid_h, c.backbone.grid_w = 2, 4, 4   # 32 tokens
        c.backbone.embed_dim = 64
        c.treat.out_dim = 16
        c.predictor.width, c.predictor.depth, c.predictor.num_heads = 64, 2, 4
        c.thickness.out_h, c.thickness.out_w, c.thickness.hidden = 5, 8, 32
        c.__post_init__()        # 重新推導衍生值 (in_dim/dt_dim/cond_dim)
        return c
