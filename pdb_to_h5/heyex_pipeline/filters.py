"""Macular cube qualification + eye-visit tie-breaker (CLAUDE_TASK.md D.3).

A series is a macular cube iff ALL of:
    n_bscans >= 15
    horizontal raster: |posY1 - posY2| < 0.05 for every B-scan
    symmetric X:       |posX1 + posX2| < 0.05 for every B-scan
    5.0 <= BScan_length_mm <= 9.0
    |centerX_deg| < 5 and |centerY_deg| < 5
    type_hex != 0x4000275d   (not OCTA)
    (bscan_h, bscan_w) in {(496, 384), (496, 512), (496, 768)}
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from heyex_pipeline.coordinates import BScanMeta, MM_PER_DEG

# -------- constants (keep in sync with CLAUDE_TASK.md D.3) --------

MIN_N_BSCANS = 15
MAX_HORIZONTAL_SLOPE_DEG = 0.05
MAX_X_ASYMMETRY_DEG = 0.05
MIN_BSCAN_LEN_MM = 5.0
MAX_BSCAN_LEN_MM = 9.0
MAX_CENTER_OFFSET_DEG = 5.0
ALLOWED_BSCAN_SHAPES: frozenset[tuple[int, int]] = frozenset({
    (496, 384),
    (496, 512),
    (496, 768),
})
OCTA_TYPE_HEX = 0x4000275D


@dataclass(frozen=True)
class MacularCubeVerdict:
    is_macular: bool
    reason: str  # "" when is_macular=True

    def __bool__(self) -> bool:
        return self.is_macular


def bscan_length_mm(meta: BScanMeta) -> float:
    """Euclidean length of a B-scan line in mm (Gullstrand deg -> mm)."""
    dx = meta.pos_x2 - meta.pos_x1
    dy = meta.pos_y2 - meta.pos_y1
    return math.hypot(dx, dy) * MM_PER_DEG


def is_macular_cube(
    bscans: list[BScanMeta],
    type_hex: int,
    bscan_h: int,
    bscan_w: int,
    center_x_deg: float | None = None,
    center_y_deg: float | None = None,
) -> MacularCubeVerdict:
    """Evaluate a series for macular-cube eligibility.

    Args:
        bscans: all B-scan metadata for this series.
        type_hex: image type ID of the OCT image entries (to detect OCTA).
        bscan_h, bscan_w: B-scan depth/A-scan dimensions.
        center_x_deg, center_y_deg: optional explicit scan center in degrees;
            falls back to the first B-scan's center_x / center_y when None.

    Returns:
        MacularCubeVerdict. `reason` names the first failed check.
    """
    if not bscans:
        return MacularCubeVerdict(False, "no_bscans")

    n = len(bscans)
    if n < MIN_N_BSCANS:
        return MacularCubeVerdict(False, f"insufficient_bscans:{n}")

    if type_hex == OCTA_TYPE_HEX:
        return MacularCubeVerdict(False, "octa_not_macular")

    if (int(bscan_h), int(bscan_w)) not in ALLOWED_BSCAN_SHAPES:
        return MacularCubeVerdict(False, f"unsupported_scan_pattern:{bscan_h}x{bscan_w}")

    for m in bscans:
        if abs(m.pos_y1 - m.pos_y2) >= MAX_HORIZONTAL_SLOPE_DEG:
            return MacularCubeVerdict(False, "not_horizontal_raster")
        if abs(m.pos_x1 + m.pos_x2) >= MAX_X_ASYMMETRY_DEG:
            return MacularCubeVerdict(False, "asymmetric_x")

    # B-scan length is a series-level property; check the first one.
    first_len_mm = bscan_length_mm(bscans[0])
    if not (MIN_BSCAN_LEN_MM <= first_len_mm <= MAX_BSCAN_LEN_MM):
        return MacularCubeVerdict(
            False, f"bscan_length_out_of_range:{first_len_mm:.2f}mm"
        )

    if center_x_deg is not None:
        cx = center_x_deg
    else:
        cx = sum(0.5 * (m.pos_x1 + m.pos_x2) for m in bscans) / n
    if center_y_deg is not None:
        cy = center_y_deg
    else:
        cy = sum(0.5 * (m.pos_y1 + m.pos_y2) for m in bscans) / n
    if abs(cx) >= MAX_CENTER_OFFSET_DEG or abs(cy) >= MAX_CENTER_OFFSET_DEG:
        return MacularCubeVerdict(
            False, f"off_center:({cx:.2f},{cy:.2f})"
        )

    return MacularCubeVerdict(True, "")


# ---------- eye-visit tie-breaker ----------


def eye_visit_sort_key(
    image_quality: float,
    valid_ascan_ratio: float,
    acquisition_time_utc: str,
    series_id: int,
) -> tuple:
    """Decreasing desirability sort key (smaller is better).

    (-image_quality, -valid_ascan_ratio, acquisition_time, series_id)
    """
    return (
        -float(image_quality),
        -float(valid_ascan_ratio),
        acquisition_time_utc or "",
        int(series_id),
    )
