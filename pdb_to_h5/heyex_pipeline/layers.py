"""Retinal layer segmentation parsing and validity helpers.

SegType semantics (human-verified on B-scan overlay; see CLAUDE_TASK.md D.1):
    5 -> ILM        (Internal Limiting Membrane)
    2 -> RPE_BM     (RPE posterior / BM complex; HEYEX mislabels as "BM")
    7 -> BM_true    (true Bruch's membrane; Advanced RPE module only,
                     typically 100% sentinel in this dataset)

Each SEG chunk (entry type 0x2723) has a 36-byte header:
    u0:uint32, index:uint32, segType:uint32, size:uint32, padding:uint32[5]
followed by `size` float32 y-pixel coordinates. FLT_MAX marks invalid.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

LAYER_NAMES: dict[int, str] = {
    5: "ILM",
    2: "RPE_BM",
    7: "BM_true",
}

CANONICAL_ORDER: tuple[int, ...] = (5, 2, 7)

SEG_HEADER_FORMAT = "<IIII5I"
SEG_HEADER_SIZE = struct.calcsize(SEG_HEADER_FORMAT)  # 36
assert SEG_HEADER_SIZE == 36

INVALID_FLOAT = np.float32(3.4028234663852886e38)


@dataclass(frozen=True)
class SegChunk:
    seg_index: int
    seg_type: int
    boundary: np.ndarray  # float32, FLT_MAX replaced with NaN

    @property
    def layer_name(self) -> str:
        return LAYER_NAMES.get(self.seg_type, f"Type{self.seg_type}")

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.boundary)

    @property
    def valid_count(self) -> int:
        return int(self.valid_mask.sum())


def parse_seg_chunk(payload: bytes) -> SegChunk | None:
    """Parse a SEG chunk payload (content bytes after data_content_offset).

    Returns None if payload is too short to hold a header.
    FLT_MAX sentinels are converted to NaN.
    """
    if len(payload) < SEG_HEADER_SIZE:
        return None
    _u0, seg_index, seg_type, size = struct.unpack_from(
        SEG_HEADER_FORMAT, payload, 0
    )[:4]
    max_elems = (len(payload) - SEG_HEADER_SIZE) // 4
    n = min(size, max_elems)
    raw = np.frombuffer(payload, dtype="<f4", count=n, offset=SEG_HEADER_SIZE)
    boundary = raw.astype(np.float32).copy()
    boundary[boundary >= INVALID_FLOAT] = np.nan
    return SegChunk(seg_index=int(seg_index), seg_type=int(seg_type), boundary=boundary)


def segmentation_types_available(seg_types_present: set[int] | dict) -> str:
    """Return comma-separated canonical list for HDF5 attribute.

    e.g. {5, 2} -> "ILM,RPE_BM"
         {5, 2, 7} -> "ILM,RPE_BM,BM_true"
    """
    if isinstance(seg_types_present, dict):
        keys = set(seg_types_present.keys())
    else:
        keys = set(seg_types_present)
    return ",".join(LAYER_NAMES[t] for t in CANONICAL_ORDER if t in keys)


def has_any_valid(boundary: np.ndarray) -> bool:
    """True if the boundary has at least one finite (non-NaN) value."""
    return bool(np.isfinite(boundary).any())


def pack_layers(
    chunks_by_type: dict[int, SegChunk],
    n_ascans: int,
) -> dict[str, np.ndarray]:
    """Assemble a single B-scan's layer arrays keyed by canonical name.

    Each output is a (n_ascans,) float32 array with NaN where invalid or
    missing. Returns only keys that are present AND have at least one
    valid A-scan.
    """
    out: dict[str, np.ndarray] = {}
    for seg_type in CANONICAL_ORDER:
        chunk = chunks_by_type.get(seg_type)
        if chunk is None:
            continue
        if not has_any_valid(chunk.boundary):
            continue
        arr = np.full(n_ascans, np.nan, dtype=np.float32)
        m = min(n_ascans, chunk.boundary.shape[0])
        arr[:m] = chunk.boundary[:m]
        out[LAYER_NAMES[seg_type]] = arr
    return out
