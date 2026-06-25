"""forecast_c.model — 最小版模型（torch）。

凍結 encoder + 一步治療條件 predictor + 厚度頭，target 由同一凍結 encoder 編未來（★免 EMA）。
"""
from .encoder import FrozenEncoder, DummyEncoder, build_octcube_encoder
from .treatment import TreatmentEncoder
from .predictor import OneStepPredictor, AdaLNZeroBlock, FiLMZeroBlock, make_cond_block
from .thickness import ThicknessHead
from .losses import predict_loss
from .forecast_model import ForecastModel, fourier_features
from . import baselines

__all__ = [
    "FrozenEncoder", "DummyEncoder", "build_octcube_encoder",
    "TreatmentEncoder", "OneStepPredictor", "AdaLNZeroBlock", "FiLMZeroBlock",
    "make_cond_block", "ThicknessHead", "predict_loss", "fourier_features",
    "ForecastModel", "baselines",
]
