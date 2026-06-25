"""Laterality (OD / OS) resolution.

Priority order (CLAUDE_TASK.md D.2):
    1. EyeData.eyeSide byte ('R' -> OD, 'L' -> OS)
    2. Patient-level metadata from .pdb (if eyeSide missing)
    3. Fail -> 'undetermined_laterality' (caller records hard failure)

NEVER infer from centerX_deg or seriesID.
"""

from __future__ import annotations

from typing import Literal

Laterality = Literal["OD", "OS"]


def laterality_from_eye_side(eye_side: str | int | bytes | None) -> Laterality | None:
    """Map a raw EyeData.eyeSide value to 'OD' | 'OS' | None.

    Accepts:
        - single-char str ('R' / 'L')
        - single byte int (ord('R') / ord('L'))
        - 1-byte bytes object (b'R' / b'L')
    Anything else (including 0x00, None, unknown codes) returns None.
    """
    if eye_side is None:
        return None
    if isinstance(eye_side, bytes):
        if len(eye_side) == 0:
            return None
        eye_side = eye_side[0]
    if isinstance(eye_side, int):
        if eye_side == ord("R"):
            return "OD"
        if eye_side == ord("L"):
            return "OS"
        return None
    if isinstance(eye_side, str):
        s = eye_side.strip().upper()
        if s == "R":
            return "OD"
        if s == "L":
            return "OS"
        return None
    return None


def resolve_laterality(
    eye_data: dict | None,
    patient_meta: dict | None = None,
) -> Laterality | None:
    """Resolve laterality using the strict priority order from D.2.

    Args:
        eye_data: parsed EyeData dict (expects key 'eyeSide'), or None.
        patient_meta: parsed patient-level metadata dict (expects key
            'eyeSide' or 'laterality'), or None.

    Returns:
        'OD', 'OS', or None if undetermined.
    """
    if eye_data is not None:
        result = laterality_from_eye_side(eye_data.get("eyeSide"))
        if result is not None:
            return result

    if patient_meta is not None:
        for key in ("eyeSide", "laterality"):
            result = laterality_from_eye_side(patient_meta.get(key))
            if result is not None:
                return result

    return None
