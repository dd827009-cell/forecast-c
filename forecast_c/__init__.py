"""forecast_c — 設計 C「治療條件化 OCT latent 預測」最小版（完全白紙重寫）。

子套件:
  config          單一真相超參（純 dataclass，無 torch）
  census          A-1 普查（純 CPU/numpy，無 torch）— 唯一現在可跑的交付
  model           凍結 encoder + 一步 predictor + 厚度頭（torch）
  data            配對/縱向 dataset 介面（torch；L40 接真資料）
  train           Phase 2 訓練入口 + 存亡評估
  phase1 / phase3 之後階段的介面 stub（多教師蒸餾 / SwinUNETR 解碼）

頂層只匯出純 dataclass config，故 `import forecast_c` 不需要 torch
（census 可在任何機器跑）。需要模型時再 `from forecast_c.model import ...`。
"""
from .config import (BackboneSpec, TreatmentEncoderConfig, PredictorConfig,
                     ThicknessHeadConfig, LossConfig, ForecastConfig)

__all__ = [
    "BackboneSpec", "TreatmentEncoderConfig", "PredictorConfig",
    "ThicknessHeadConfig", "LossConfig", "ForecastConfig",
]
