"""B-scan ordering, BScanMetaData parsing, and IR/SLO spatial mapping.

Authoritative offsets (CLAUDE_TASK.md Part H + existing/export_e2e_csv.py).
DO NOT use existing/export_ir_localizer.py's offsets - they are shifted +4.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# SLO FOV lookup by SLO width (pixels)
SLO_FOV_MAP: dict[int, float] = {
    768: 30.0,
    384: 15.0,
}

# Gullstrand eye model: 1 deg ~ 0.288 mm on retina (per existing/export_ir_localizer.py).
MM_PER_DEG: float = 0.288

# BScanMetaData offsets (0x2714 payload)
OFF_IMG_SIZE_X = 0x04
OFF_IMG_SIZE_Y = 0x08
OFF_POS_X1 = 0x0C
OFF_POS_Y1 = 0x10
OFF_POS_X2 = 0x14
OFF_POS_Y2 = 0x18
OFF_SCALE_Y = 0x24
OFF_IMG_SIZE_WIDTH = 0x3C
OFF_NUM_IMAGES = 0x40
OFF_AKT_IMAGE = 0x44
OFF_SCANTYPE = 0x48
OFF_CENTER_X = 0x4C
OFF_CENTER_Y = 0x50
OFF_ACQ_TIME = 0x58
OFF_IMG_QUALITY = 0x9C
BSCAN_META_MIN_SIZE = 0x60  # 96 bytes; imageQuality requires >= 0xA0


@dataclass(frozen=True)
class BScanMeta:
    img_size_x: int          # depth pixels (~496)
    img_size_y: int          # unreliable; do not trust
    pos_x1: float            # deg
    pos_y1: float            # deg
    pos_x2: float            # deg
    pos_y2: float            # deg
    scale_y: float           # mm per axial pixel
    img_size_width: int      # A-scans (~512)
    num_images: int          # B-scans in series
    akt_image: int           # Heidelberg B-scan index (use as D-axis index)
    scantype: int            # 0 unknown, 1 line/star, 2 circle
    center_x: float          # deg
    center_y: float          # deg
    acquisition_time: int    # FILETIME (100-ns ticks since 1601-01-01 UTC)
    image_quality: float     # 0 if not present in payload


def parse_bscan_metadata(payload: bytes) -> BScanMeta | None:
    """Parse BScanMetaData payload. Returns None if too short."""
    if len(payload) < BSCAN_META_MIN_SIZE:
        return None
    img_q = 0.0
    if len(payload) >= 0xA0:
        img_q = struct.unpack_from("<f", payload, OFF_IMG_QUALITY)[0]
    return BScanMeta(
        img_size_x=struct.unpack_from("<I", payload, OFF_IMG_SIZE_X)[0],
        img_size_y=struct.unpack_from("<I", payload, OFF_IMG_SIZE_Y)[0],
        pos_x1=struct.unpack_from("<f", payload, OFF_POS_X1)[0],
        pos_y1=struct.unpack_from("<f", payload, OFF_POS_Y1)[0],
        pos_x2=struct.unpack_from("<f", payload, OFF_POS_X2)[0],
        pos_y2=struct.unpack_from("<f", payload, OFF_POS_Y2)[0],
        scale_y=struct.unpack_from("<f", payload, OFF_SCALE_Y)[0],
        img_size_width=struct.unpack_from("<I", payload, OFF_IMG_SIZE_WIDTH)[0],
        num_images=struct.unpack_from("<I", payload, OFF_NUM_IMAGES)[0],
        akt_image=struct.unpack_from("<I", payload, OFF_AKT_IMAGE)[0],
        scantype=struct.unpack_from("<I", payload, OFF_SCANTYPE)[0],
        center_x=struct.unpack_from("<f", payload, OFF_CENTER_X)[0],
        center_y=struct.unpack_from("<f", payload, OFF_CENTER_Y)[0],
        acquisition_time=struct.unpack_from("<Q", payload, OFF_ACQ_TIME)[0],
        image_quality=img_q,
    )


def bscan_index(meta: BScanMeta) -> int:
    """Canonical B-scan index within a series.

    Uses aktImage (offset 0x44). DO NOT use imageID // 2 - that fails
    whenever seg / SLO entries reshuffle imageID spacing.
    """
    return int(meta.akt_image)


def sort_bscans_superior_to_inferior(metas: list[BScanMeta]) -> list[BScanMeta]:
    """Order B-scans along the D-axis: superior -> inferior.

    Sort key is posY1_deg descending (superior == larger +Y in deg space;
    deg_to_slo_pixel flips Y to IR row). Ties are broken by aktImage
    ascending so the order is deterministic.
    """
    return sorted(metas, key=lambda m: (-float(m.pos_y1), int(m.akt_image)))


def deg_to_slo_pixel(
    x_deg: float,
    y_deg: float,
    fov: float,
    slo_w: int,
    slo_h: int,
) -> tuple[float, float]:
    """Convert (deg, deg) to (SLO px x, SLO px y). Y axis is flipped."""
    if fov <= 0:
        return float("nan"), float("nan")
    px = (x_deg / fov + 0.5) * slo_w
    py = (0.5 - y_deg / fov) * slo_h
    return px, py


def fov_from_slo_width(slo_w: int) -> float:
    return SLO_FOV_MAP.get(int(slo_w), 0.0)


def per_ascan_ir_positions(
    pos_x1: float,
    pos_y1: float,
    pos_x2: float,
    pos_y2: float,
    n_ascans: int,
    slo_w: int,
    slo_h: int,
    fov: float | None = None,
) -> np.ndarray:
    """Compute per-A-scan IR pixel coordinates for one B-scan.

    Returns array of shape (n_ascans, 2), dtype float32, where [..., 0]
    is SLO x-pixel and [..., 1] is SLO y-pixel. NaN if fov == 0.
    """
    if fov is None:
        fov = fov_from_slo_width(slo_w)
    out = np.full((n_ascans, 2), np.nan, dtype=np.float32)
    if fov <= 0 or n_ascans <= 0:
        return out
    if n_ascans == 1:
        t = np.array([0.5], dtype=np.float64)
    else:
        t = np.linspace(0.0, 1.0, n_ascans, dtype=np.float64)
    x_deg = pos_x1 + t * (pos_x2 - pos_x1)
    y_deg = pos_y1 + t * (pos_y2 - pos_y1)
    out[:, 0] = (x_deg / fov + 0.5) * slo_w
    out[:, 1] = (0.5 - y_deg / fov) * slo_h
    return out
