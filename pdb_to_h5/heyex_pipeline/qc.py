"""Sanity checks + QC flag generation (CLAUDE_TASK.md E.3).

Hard failures cause us to skip the sample and write a failures.jsonl entry:
    undetermined_laterality
    insufficient_bscans
    unsupported_scan_pattern
    not_macular_volume
    missing_layer_segmentation
    qc_fail_rpe_below_ilm     (> 5% A-scans violate rpe_bm_y > ilm_y)
    inverted_h_axis           (median rpe_bm_y < ilm_y in central strip)
    corrupt_pixel_data

Soft flags still produce an HDF5 (appended comma-separated to `flags`):
    low_layer_coverage     (valid_ascan_ratio < 0.70)
    low_quality            (image_quality < 15)
    irregular_bscan_spacing (spacing CV > 0.1)
    extreme_thickness      (median thickness not in 50..150 px)
    missing_ir
    fovea_undetectable
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---- thresholds (keep in sync with CLAUDE_TASK.md E.3) -------------------

RPE_BELOW_ILM_VIOLATION_RATIO_MAX = 0.05
LOW_LAYER_COVERAGE_THRESHOLD = 0.70
LOW_QUALITY_THRESHOLD = 15.0
IRREGULAR_SPACING_CV = 0.10
THICKNESS_MIN_PX = 50.0
THICKNESS_MAX_PX = 150.0


HARD_FAIL_REASONS = frozenset({
    "undetermined_laterality",
    "insufficient_bscans",
    "unsupported_scan_pattern",
    "not_macular_volume",
    "missing_layer_segmentation",
    "qc_fail_rpe_below_ilm",
    "inverted_h_axis",
    "corrupt_pixel_data",
})

SOFT_FLAG_NAMES = (
    "low_layer_coverage",
    "low_quality",
    "irregular_bscan_spacing",
    "extreme_thickness",
    "missing_ir",
    "fovea_undetectable",
)


@dataclass
class QCResult:
    hard_fail: str | None = None
    soft_flags: list[str] = field(default_factory=list)

    @property
    def is_hard_fail(self) -> bool:
        return self.hard_fail is not None

    def flags_attr(self) -> str:
        """Comma-separated list for the HDF5 `flags` attribute."""
        return ",".join(self.soft_flags)


def check_h_axis(
    ilm_y: np.ndarray,      # (D, W) float32, NaN where invalid
    rpe_bm_y: np.ndarray,   # (D, W) float32, NaN where invalid
) -> tuple[str | None, float]:
    """Validate that rpe_bm_y > ilm_y holds on (nearly) every valid A-scan.

    Returns (hard_fail_reason_or_None, violation_ratio).
    """
    both_valid = np.isfinite(ilm_y) & np.isfinite(rpe_bm_y)
    n_valid = int(both_valid.sum())
    if n_valid == 0:
        return "missing_layer_segmentation", 0.0

    violations = both_valid & (rpe_bm_y <= ilm_y)
    n_viol = int(violations.sum())
    ratio = n_viol / n_valid

    # Median relation in central half-width tests full axis inversion.
    w = ilm_y.shape[1]
    lo = int(w * 0.25)
    hi = int(w * 0.75)
    if hi > lo:
        d_center = ilm_y.shape[0] // 2
        center_mask = both_valid[d_center, lo:hi]
        if center_mask.any():
            ilm_med = float(np.nanmedian(ilm_y[d_center, lo:hi][center_mask]))
            rpe_med = float(np.nanmedian(rpe_bm_y[d_center, lo:hi][center_mask]))
            if rpe_med < ilm_med:
                return "inverted_h_axis", ratio

    if ratio > RPE_BELOW_ILM_VIOLATION_RATIO_MAX:
        return "qc_fail_rpe_below_ilm", ratio
    return None, ratio


def bscan_spacing_cv(bscan_y_deg: np.ndarray) -> float:
    """Coefficient of variation of |delta posY1| between consecutive B-scans.

    Returns 0 when n < 2 (not enough data to judge irregularity).
    """
    y = np.asarray(bscan_y_deg, dtype=np.float64)
    if y.size < 2:
        return 0.0
    diffs = np.abs(np.diff(y))
    mean = diffs.mean()
    if mean <= 0:
        return float("inf")  # all identical -> treat as degenerate
    return float(diffs.std() / mean)


def compute_soft_flags(
    *,
    valid_ascan_ratio: float,
    image_quality: float,
    bscan_spacing_deg: np.ndarray | None,
    median_thickness_px: float | None,
    has_ir: bool,
    fovea_valid: bool,
) -> list[str]:
    flags: list[str] = []

    if valid_ascan_ratio < LOW_LAYER_COVERAGE_THRESHOLD:
        flags.append("low_layer_coverage")

    if image_quality < LOW_QUALITY_THRESHOLD:
        flags.append("low_quality")

    if bscan_spacing_deg is not None and len(bscan_spacing_deg) >= 2:
        cv = bscan_spacing_cv(bscan_spacing_deg)
        if cv > IRREGULAR_SPACING_CV:
            flags.append("irregular_bscan_spacing")

    if median_thickness_px is not None and np.isfinite(median_thickness_px):
        if not (THICKNESS_MIN_PX <= median_thickness_px <= THICKNESS_MAX_PX):
            flags.append("extreme_thickness")

    if not has_ir:
        flags.append("missing_ir")

    if not fovea_valid:
        flags.append("fovea_undetectable")

    return flags
