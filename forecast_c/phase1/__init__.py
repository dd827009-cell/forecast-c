"""forecast_c.phase1 — 表徵階段（多教師蒸餾 + 3D adapter + 空間 JEPA）。

①a 2D per-slice 學生 = 多教師蒸餾（DINOv2+RETFound+MedSAM2）+ MIM
     → **vendored 楊瀚博 FoundMIM**（`foundmim/`），OCT 域接合在 `distill.py`。
①b 跨切片 = 3D adapter（**vendored 吳韋論 Adapter3D**，`adapter3d.py`，depth-only 3×1×1）
     插進凍結 2D 學生，用**空間 JEPA**（`spatial_jepa.py`，設計 C 原創）訓 adapter。

⚠️ 元件來自兩篇前輩 code；**JEPA 訓練法（空間+時間）是設計 C 原創**，兩篇都沒有。
"""
from .adapter3d import Adapter3D, AdaptedBlock, inject_adapter3d
from .teachers import TeacherFeatureExtractor
from .distill import (build_foundmim_for_oct, OCT_TEACHERS, DEFAULT_OCT_TEACHERS,
                      STUDENT_FACTORY)
from .spatial_jepa import SpatialJEPA, slice_mask

__all__ = [
    "Adapter3D", "AdaptedBlock", "inject_adapter3d",
    "build_foundmim_for_oct", "OCT_TEACHERS", "DEFAULT_OCT_TEACHERS",
    "TeacherFeatureExtractor", "STUDENT_FACTORY",
    "SpatialJEPA", "slice_mask",
]
