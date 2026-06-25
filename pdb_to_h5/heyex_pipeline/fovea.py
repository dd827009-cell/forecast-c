"""Fovea center estimation via thickness minimum (CLAUDE_TASK.md E.7).

Algorithm:
    1. thickness(d, w) = rpe_bm_y[d, w] - ilm_y[d, w] for valid A-scans
    2. Central window = D//2 +/- 2 (5 central B-scans)
    3. In central window, find (d*, w*) minimising thickness subject to a
       3x3 valid_mask neighbourhood with valid_ratio > 0.9
    4. Return ascan_pos_ir[d*, w*] (x, y in IR pixels)
    5. If no valid candidate: return (NaN, NaN).
"""

from __future__ import annotations

import numpy as np

CENTRAL_HALF_WINDOW = 2
NEIGHBORHOOD_SIZE = 3
NEIGHBORHOOD_MIN_VALID_RATIO = 0.9


def estimate_fovea_ir_xy(
    ilm_y: np.ndarray,          # (D, W) float32, NaN where invalid
    rpe_bm_y: np.ndarray,       # (D, W) float32, NaN where invalid
    valid_mask: np.ndarray,     # (D, W) bool
    ascan_pos_ir: np.ndarray,   # (D, W, 2) float32
    n_bscans: int,
) -> tuple[float, float]:
    """Return (x, y) IR-pixel fovea estimate, or (nan, nan) if not detectable.

    `n_bscans` is passed explicitly for clarity; must equal `ilm_y.shape[0]`.
    """
    if n_bscans != ilm_y.shape[0] or ilm_y.shape != rpe_bm_y.shape:
        return (float("nan"), float("nan"))

    d = n_bscans
    if d < 1:
        return (float("nan"), float("nan"))

    # Thickness only where valid
    thickness = np.where(valid_mask, rpe_bm_y - ilm_y, np.nan).astype(np.float32)

    # Central D window: D//2 +/- CENTRAL_HALF_WINDOW (clipped)
    center = d // 2
    d_lo = max(0, center - CENTRAL_HALF_WINDOW)
    d_hi = min(d, center + CENTRAL_HALF_WINDOW + 1)
    if d_lo >= d_hi:
        return (float("nan"), float("nan"))

    # 3x3 neighbourhood valid-ratio filter (integer summed-area via
    # cumulative sum is overkill for a 5xW window; straight indexing is fine).
    h = NEIGHBORHOOD_SIZE // 2  # 1
    min_valid = NEIGHBORHOOD_MIN_VALID_RATIO * (NEIGHBORHOOD_SIZE ** 2)

    best_d = -1
    best_w = -1
    best_thickness = np.inf

    w_dim = ilm_y.shape[1]
    if w_dim == 0:
        return (float("nan"), float("nan"))

    for di in range(d_lo, d_hi):
        for wi in range(w_dim):
            t = thickness[di, wi]
            if not np.isfinite(t):
                continue
            dd0 = max(0, di - h)
            dd1 = min(d, di + h + 1)
            ww0 = max(0, wi - h)
            ww1 = min(w_dim, wi + h + 1)
            neigh = valid_mask[dd0:dd1, ww0:ww1]
            # Pad-denominator: use full 3x3 so edge candidates need most of the
            # available neighbours to be valid.
            if neigh.sum() < min_valid:
                continue
            if t < best_thickness:
                best_thickness = float(t)
                best_d = di
                best_w = wi

    if best_d < 0:
        return (float("nan"), float("nan"))

    pos = ascan_pos_ir[best_d, best_w]
    x = float(pos[0])
    y = float(pos[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return (float("nan"), float("nan"))
    return (x, y)
