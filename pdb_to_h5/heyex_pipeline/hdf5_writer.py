"""HDF5 writer matching CLAUDE_TASK.md E.3 schema.

One file per successful eye-visit. Output path layout:

    {out_root}/{h[:2]}/{h[2:4]}/{patient_id}/{visit_id}_{laterality}.h5
    h = sha1(patient_id.encode()).hexdigest()

Uses atomic write: writes to `{path}.partial`, then renames into place only
after the file is fully closed. Crash mid-write leaves no half file behind.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from heyex_pipeline.version import parser_version

H5_SUFFIX = ".h5"


@dataclass
class HDF5Sample:
    """Fully assembled sample ready for on-disk write.

    Shapes and dtypes match E.3 exactly. Optional datasets can be None.
    """

    # Root identifiers
    patient_id: str
    visit_date: str
    visit_id: str
    acquisition_time_utc: str
    laterality: str  # "OD" | "OS"

    # Volume (D, H, W) float32 — |fp16| linear reflectance, raw signal.
    # Display = sqrt(volume) then percentile-normalize.
    volume: np.ndarray
    # IR (H_ir, W_ir) uint8 — use shape (0, 0) when absent
    ir: np.ndarray

    # Layers (D, W) float32 with NaN where invalid
    ilm_y: np.ndarray
    rpe_bm_y: np.ndarray
    bm_true_y: np.ndarray | None  # only when segType 7 has any valid values

    # (D, W) bool and (D, W, 2) float32 (use (0,0,0) shape if no IR)
    valid_ascan_mask: np.ndarray
    ascan_pos_ir: np.ndarray

    # Optional
    image_quality_per_bscan: np.ndarray | None  # (D,) float32 or None
    line_scans: list[dict[str, Any]] = field(default_factory=list)
    # Each dict: {'pixels': (H,W) uint8, 'pattern': str,
    #             'ir_x1','ir_y1','ir_x2','ir_y2','image_quality'}

    # Scalar root attributes
    scale_axial_um_per_px: float = 0.0
    scale_lateral_mm_per_px: float = 0.0
    scale_bscan_spacing_mm: float = 0.0
    bscan_spacing_deg_per_index: float = 0.0
    image_quality: float = 0.0
    valid_ascan_count: int = 0
    valid_ascan_ratio: float = 0.0
    segmentation_types_available: str = ""
    has_ir: bool = False
    has_line_scans: bool = False
    n_line_scans: int = 0
    fovea_ir_x: float = float("nan")
    fovea_ir_y: float = float("nan")
    source_sdb_path: str = ""
    source_edb_path: str = ""
    source_pdb_path: str = ""
    flags: str = ""

    # Demographics (PHI — keep only what training needs)
    sex: str = ""                      # "Male" / "Female" / "Unknown" / ""
    birth_date: str = ""               # "YYYY-MM-DD" or ""
    age_at_visit_years: float = float("nan")


def h5_relative_path(patient_id: str, visit_id: str, laterality: str) -> Path:
    """Deterministic on-disk layout, relative to the output root."""
    h = hashlib.sha1(patient_id.encode("utf-8")).hexdigest()
    return Path(h[:2]) / h[2:4] / patient_id / f"{visit_id}_{laterality}{H5_SUFFIX}"


def write_sample(sample: HDF5Sample, out_root: str | os.PathLike) -> Path:
    """Write a single sample HDF5. Returns the final on-disk path.

    Atomic: writes to `<path>.partial` first, fsyncs, then renames.
    """
    import h5py

    out_root = Path(out_root)
    rel = h5_relative_path(sample.patient_id, sample.visit_id, sample.laterality)
    final_path = out_root / rel
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".partial")

    n_bscans, bscan_h, bscan_w = sample.volume.shape

    with h5py.File(tmp_path, "w") as h5:
        # ---- Datasets ----
        vol_ds = h5.create_dataset(
            "volume",
            data=np.ascontiguousarray(sample.volume, dtype=np.float32),
            chunks=(1, bscan_h, bscan_w),
            compression="lzf",
        )
        # Convention note: raw |fp16| linear reflectance. For display,
        # apply sqrt() then percentile-normalise (see Heidelberg HEYEX).
        vol_ds.attrs["encoding"] = "abs_float16_as_float32"
        vol_ds.attrs["display_transform"] = "sqrt_then_percentile_norm"
        h5.create_dataset(
            "ir",
            data=np.ascontiguousarray(sample.ir, dtype=np.uint8),
            compression="lzf" if sample.ir.size > 0 else None,
        )
        h5.create_dataset(
            "ilm_y",
            data=np.ascontiguousarray(sample.ilm_y, dtype=np.float32),
        )
        h5.create_dataset(
            "rpe_bm_y",
            data=np.ascontiguousarray(sample.rpe_bm_y, dtype=np.float32),
        )
        if sample.bm_true_y is not None:
            h5.create_dataset(
                "bm_true_y",
                data=np.ascontiguousarray(sample.bm_true_y, dtype=np.float32),
            )
        h5.create_dataset(
            "valid_ascan_mask",
            data=np.ascontiguousarray(sample.valid_ascan_mask, dtype=bool),
        )
        h5.create_dataset(
            "ascan_pos_ir",
            data=np.ascontiguousarray(sample.ascan_pos_ir, dtype=np.float32),
        )
        if sample.image_quality_per_bscan is not None:
            h5.create_dataset(
                "image_quality_per_bscan",
                data=np.ascontiguousarray(
                    sample.image_quality_per_bscan, dtype=np.float32
                ),
            )

        # ---- /line_scans/ group ----
        line_grp = h5.create_group("line_scans")
        for i, ls in enumerate(sample.line_scans):
            ds = line_grp.create_dataset(
                f"scan_{i:03d}",
                data=np.ascontiguousarray(ls["pixels"], dtype=np.uint8),
                compression="lzf",
            )
            ds.attrs["pattern"] = str(ls.get("pattern", ""))
            ds.attrs["ir_x1"] = np.float32(ls.get("ir_x1", float("nan")))
            ds.attrs["ir_y1"] = np.float32(ls.get("ir_y1", float("nan")))
            ds.attrs["ir_x2"] = np.float32(ls.get("ir_x2", float("nan")))
            ds.attrs["ir_y2"] = np.float32(ls.get("ir_y2", float("nan")))
            ds.attrs["image_quality"] = np.float32(
                ls.get("image_quality", float("nan"))
            )

        # ---- Root attributes ----
        attrs = h5.attrs
        attrs["patient_id"] = sample.patient_id
        attrs["visit_date"] = sample.visit_date
        attrs["visit_id"] = sample.visit_id
        attrs["acquisition_time_utc"] = sample.acquisition_time_utc
        attrs["laterality"] = sample.laterality
        attrs["n_bscans"] = np.int32(n_bscans)
        attrs["bscan_height"] = np.int32(bscan_h)
        attrs["bscan_width"] = np.int32(bscan_w)
        attrs["scale_axial_um_per_px"] = np.float32(sample.scale_axial_um_per_px)
        attrs["scale_lateral_mm_per_px"] = np.float32(sample.scale_lateral_mm_per_px)
        attrs["scale_bscan_spacing_mm"] = np.float32(sample.scale_bscan_spacing_mm)
        attrs["bscan_spacing_deg_per_index"] = np.float32(
            sample.bscan_spacing_deg_per_index
        )
        attrs["image_quality"] = np.float32(sample.image_quality)
        attrs["valid_ascan_count"] = np.int32(sample.valid_ascan_count)
        attrs["valid_ascan_ratio"] = np.float32(sample.valid_ascan_ratio)
        attrs["longitudinal_key"] = f"{sample.patient_id}::{sample.laterality}"
        attrs["visit_uid"] = f"{sample.patient_id}::{sample.visit_id}"
        attrs["segmentation_types_available"] = sample.segmentation_types_available
        attrs["has_ir"] = bool(sample.has_ir)
        attrs["has_line_scans"] = bool(sample.has_line_scans)
        attrs["n_line_scans"] = np.int32(sample.n_line_scans)
        attrs["fovea_ir_x"] = np.float32(sample.fovea_ir_x)
        attrs["fovea_ir_y"] = np.float32(sample.fovea_ir_y)
        attrs["parser_version"] = parser_version
        attrs["source_sdb_path"] = sample.source_sdb_path
        attrs["source_edb_path"] = sample.source_edb_path
        attrs["source_pdb_path"] = sample.source_pdb_path
        attrs["flags"] = sample.flags
        attrs["sex"] = sample.sex
        attrs["birth_date"] = sample.birth_date
        attrs["age_at_visit_years"] = np.float32(sample.age_at_visit_years)

    os.replace(tmp_path, final_path)
    return final_path


# ---------- idempotency helpers (used by CLI) ----------


def existing_major_matches(path: Path, current_major: int) -> bool:
    """True if `path` exists and has a parser_version with matching major."""
    if not path.exists():
        return False
    import h5py

    try:
        with h5py.File(path, "r") as h5:
            v = h5.attrs.get("parser_version", "")
            if not v:
                return False
            major = int(str(v).split(".", 1)[0])
            return major == int(current_major)
    except (OSError, ValueError):
        return False
