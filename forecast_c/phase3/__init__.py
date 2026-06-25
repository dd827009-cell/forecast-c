"""forecast_c.phase3 — 密集讀出（之後做，目前介面 stub）。

3D 特徵 → SwinUNETR 多尺度解碼 → 厚度µm + 積水分割→體積變化（這次−上次）。
前置 REG-1 配準（用 ascan_pos_ir 把回診對齊 baseline，QC 閘門）。
"""
from .decode import SwinUNETRDecoder, fluid_volume_change, REG1Registration

__all__ = ["SwinUNETRDecoder", "fluid_volume_change", "REG1Registration"]
