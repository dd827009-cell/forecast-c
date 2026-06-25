"""forecast_c.census — A-1 普查（純 CPU/numpy，無 torch）。

規格: `solutions/A1_census_spec.md`。燒 L40 前用純 CPU 算關鍵數字，決定 Stage A
可不可行 / 要不要降級。可在任何機器跑（含本機 pilot）。

模組:
  cst        厚度 / 中央 1mm CST（(rpe-ilm)*axial、NaN/層交叉無效、圈內平均）
  recovery   變乾事件 (event_step, event_observed) — 右設限、NaN→未乾
  a1_census  普查 driver：五組數字 + 決策表 → census_report.{json,md}
"""
from .cst import (thickness_um, central_subfield_thickness, cst_from_npz,
                  cst_stats, normalize_cst)
from .recovery import (is_dry, dry_sequence, recovery_event,
                       batch_recovery_events, recovery_rate)

__all__ = [
    "thickness_um", "central_subfield_thickness", "cst_from_npz", "cst_stats",
    "normalize_cst", "is_dry", "dry_sequence", "recovery_event",
    "batch_recovery_events", "recovery_rate",
]
