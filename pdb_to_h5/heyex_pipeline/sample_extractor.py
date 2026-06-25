"""Extract eye-visit samples from a `.pat` directory.

This is the Stage 3 integrator. For one `.pat`:

    1. Scan all `.sdb/.edb/.pdb` files into dir-entries.
    2. Parse patient + eye + study metadata.
    3. Group OCT entries by seriesID and qualify each series as a macular cube.
    4. Build a full `HDF5Sample` for each qualifying cube.
    5. Pick one sample per (patient_id, laterality, visit_date) via the
       eye-visit tie-breaker from filters.py.

We reuse the primitives from `existing/export_e2e_csv.py` unchanged.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from heyex_pipeline import coordinates as C
from heyex_pipeline import layers as L
from heyex_pipeline.filters import (
    MacularCubeVerdict,
    eye_visit_sort_key,
    is_macular_cube,
    MM_PER_DEG,
)
from heyex_pipeline.fovea import estimate_fovea_ir_xy
from heyex_pipeline.hdf5_writer import HDF5Sample
from heyex_pipeline.laterality import resolve_laterality
from heyex_pipeline.qc import (
    QCResult,
    check_h_axis,
    compute_soft_flags,
)

logger = logging.getLogger(__name__)

# Make the existing/ parsers importable without editing them.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXISTING = _REPO_ROOT / "existing"
if str(_EXISTING) not in sys.path:
    sys.path.insert(0, str(_EXISTING))

from export_e2e_csv import (  # noqa: E402
    DATA_ENTRY_HEADER_SIZE,
    data_content_offset,
    parse_eye_data,
    parse_patient_data,
    parse_string_list,
    parse_text_element,
    scan_all_files,
)

# Heidelberg image-type codes (from export_e2e_csv.TYPE_NAMES)
TYPE_IMAGE = 0x40000000
TYPE_OCTA = 0x4000275D
TYPE_BSCAN_META = 0x2714
TYPE_SEG = 0x2723
TYPE_EYE_DATA = 0x07
TYPE_PATIENT_DATA = 0x09
TYPE_SCAN_PATTERN = 0x232E
TYPE_PATIENT_UID = 0x34

IMAGE_HEADER_SIZE = 20


# =========================================================================
# Utilities
# =========================================================================


def filetime_to_utc_iso(ft: int | None) -> str | None:
    """Convert Heidelberg FILETIME (100-ns ticks since 1601-01-01 UTC)
    to ISO-8601 UTC, e.g. "2012-06-12T02:33:51+00:00".

    Returns None for ft == 0 / None.
    """
    if not ft:
        return None
    try:
        dt = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(
            microseconds=int(ft) // 10
        )
        return dt.isoformat()
    except (OverflowError, ValueError, OSError):
        return None


def iso_to_visit_id(iso_str: str | None) -> str | None:
    """Filesystem-safe "YYYYMMDDTHHMMSS" (no colons/dashes)."""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y%m%dT%H%M%S")
    except ValueError:
        return None


# =========================================================================
# Low-level readers (operate on open file objects from existing/parsers)
# =========================================================================


def _read_image_header(f, entry: dict) -> dict | None:
    """Read the 20-byte ImageHeader prefix of an image entry."""
    f.seek(data_content_offset(entry))
    raw = f.read(IMAGE_HEADER_SIZE)
    if len(raw) < IMAGE_HEADER_SIZE:
        return None
    _, _, img_type, _ = struct.unpack_from("BBBB", raw, 4)
    _u5, breite, hoehe = struct.unpack_from("<III", raw, 8)
    # existing/export_image_details clarifies: breite holds depth/height,
    # hoehe holds width. We keep that interpretation to stay consistent.
    return {
        "img_type": int(img_type),
        "breite": int(breite),   # axial depth (H) for B-scans, pixel rows for SLO
        "hoehe": int(hoehe),     # A-scan count (W) for B-scans, pixel cols for SLO
        "pixel_offset": entry["dataAddress"] + DATA_ENTRY_HEADER_SIZE + IMAGE_HEADER_SIZE,
    }


def _read_image_pixels(f, entry: dict, hdr: dict) -> np.ndarray | None:
    """Read raw pixel bytes of an image entry into a numpy array.

    8-bit path (img_type == 1, or subID==0): uint8 SLO / IR localizer.
    16-bit path (img_type == 32, or subID==1): Heidelberg OCT B-scan.

    Heidelberg gotcha: the image header type 32 nominally means CV_16UC1
    (uint16), but the 16-bit bytes are actually IEEE 754 half-precision
    floats encoding linear reflectance. We:
        1. read as little-endian float16
        2. zero out non-finite values (inf / NaN)
        3. take abs() (sign bit is part of the encoding's noise)
        4. cast to float32 for downstream stability

    The displayable image is `sqrt(volume)` followed by percentile
    normalisation; we leave that step to the consumer so we can keep
    the raw linear reflectance signal here.
    """
    img_type = hdr["img_type"]
    if img_type == 1:
        is_oct = False
    elif img_type == 32:
        is_oct = True
    else:
        is_oct = entry.get("subID", -1) != 0

    bpp = 2 if is_oct else 1
    need = hdr["breite"] * hdr["hoehe"] * bpp
    avail = entry["dataLength"] - IMAGE_HEADER_SIZE
    if need > avail:
        return None

    f.seek(hdr["pixel_offset"])
    buf = f.read(need)
    if len(buf) != need:
        return None

    if not is_oct:
        return np.frombuffer(buf, dtype=np.uint8).reshape(
            hdr["breite"], hdr["hoehe"]
        )

    # OCT path: float16 with non-finite scrubbing
    arr = np.frombuffer(buf, dtype="<f2").reshape(hdr["breite"], hdr["hoehe"])
    arr = arr.astype(np.float32)
    arr[~np.isfinite(arr)] = 0.0
    np.abs(arr, out=arr)
    return arr


def _read_seg_chunk(f, entry: dict) -> L.SegChunk | None:
    """Load one SEG entry via layers.parse_seg_chunk."""
    f.seek(data_content_offset(entry))
    payload = f.read(entry["dataLength"])
    return L.parse_seg_chunk(payload)


def _read_bscan_meta(f, entry: dict) -> C.BScanMeta | None:
    """Load one BScanMetaData entry via coordinates.parse_bscan_metadata."""
    f.seek(data_content_offset(entry))
    payload = f.read(min(entry["dataLength"], 256))
    return C.parse_bscan_metadata(payload)


# =========================================================================
# In-memory grouping
# =========================================================================


@dataclass
class PatientContext:
    patient_id: str
    patient_uid: str = ""
    sex: str = ""               # "Male" / "Female" / "Unknown"
    birth_date: str = ""        # "YYYY-MM-DD"
    # study_id -> set of eye sides observed at that study ('OD' / 'OS')
    eye_sides_by_study: dict[int, set] = field(default_factory=dict)
    # Populated after series bundles are collected (per-series laterality).
    laterality_by_series: dict[int, str] = field(default_factory=dict)
    scan_pattern_by_series: dict[int, str] = field(default_factory=dict)
    source_paths: dict[str, str] = field(default_factory=dict)  # ext -> path


@dataclass
class SeriesBundle:
    """All per-series entries needed to build one sample."""

    series_id: int
    study_id: int
    bscan_metas: dict[int, tuple[dict, C.BScanMeta]] = field(default_factory=dict)
    # image_id -> (dir_entry, BScanMeta)

    oct_images: dict[int, dict] = field(default_factory=dict)
    # image_id -> dir_entry for subID==1 images

    slo_image: dict | None = None  # dir_entry for subID==0 image

    seg_by_image: dict[int, dict[int, dict]] = field(default_factory=dict)
    # image_id -> {seg_type: dir_entry}

    oct_type_hex: int = TYPE_IMAGE  # differ from TYPE_OCTA for normal B-scans


def _collect_patient(entries_with_files: list[tuple[dict, str]]) -> PatientContext:
    """Pull patient-level metadata from a list of (entry, filepath)."""
    patient_id_int = None
    patient_id_str = ""
    medical_record_no = ""   # digits-only ID extracted from `surname` (CGMH convention)
    patient_uid = ""
    sex = ""
    birth_date = ""
    eye_sides_by_study: dict[int, set] = {}
    scan_pattern_by_series: dict[int, str] = {}

    source_paths: dict[str, str] = {}
    for entry, filepath in entries_with_files:
        ext = entry.get("_sourceExt", "")
        if ext and ext not in source_paths:
            source_paths[ext] = filepath
        if patient_id_int is None and entry.get("patientID", -1) >= 0:
            patient_id_int = entry["patientID"]

    # One open per file for the small per-patient entries.
    file_cache: dict[str, Any] = {}
    try:
        for entry, filepath in entries_with_files:
            if filepath not in file_cache:
                file_cache[filepath] = open(filepath, "rb")
            f = file_cache[filepath]
            t = entry["type"]
            if t == TYPE_PATIENT_DATA:
                pd = parse_patient_data(f, entry)
                if pd:
                    if not patient_id_str and pd.get("patientIdStr"):
                        patient_id_str = pd["patientIdStr"]
                    # CGMH convention: the real medical record number is in
                    # the `surname` field (e.g. "4561107-7"). Heidelberg's
                    # `patientIdStr` is empty for this site.
                    if not medical_record_no:
                        sn = pd.get("surname", "") or ""
                        digits = "".join(ch for ch in sn if ch.isdigit())
                        if digits:
                            medical_record_no = digits
                    if not sex and pd.get("sex"):
                        sex = pd["sex"]
                    if not birth_date and pd.get("birthDate"):
                        # birthDate is "YYYY-MM-DD HH:MM:SS"; keep date only.
                        bd = pd["birthDate"]
                        birth_date = bd[:10] if len(bd) >= 10 else bd
            elif t == TYPE_PATIENT_UID:
                uid = parse_text_element(f, entry)
                if uid:
                    patient_uid = uid
            elif t == TYPE_EYE_DATA:
                ed = parse_eye_data(f, entry)
                if ed is not None:
                    from heyex_pipeline.laterality import laterality_from_eye_side
                    side = laterality_from_eye_side(ed.get("eyeSide"))
                    if side is not None:
                        eye_sides_by_study.setdefault(
                            entry["studyID"], set()
                        ).add(side)
            elif t == TYPE_SCAN_PATTERN:
                strings = parse_string_list(f, entry)
                if strings:
                    scan_pattern_by_series[entry["seriesID"]] = "; ".join(strings)
    finally:
        for f in file_cache.values():
            f.close()

    # Identity priority:
    #   1. medical_record_no (CGMH-style, from `surname`)
    #   2. Heidelberg patient_id_str (standard, usually empty for CGMH)
    #   3. integer patientID fallback
    final_id = medical_record_no or patient_id_str or (
        str(patient_id_int) if patient_id_int is not None else "unknown"
    )

    return PatientContext(
        patient_id=final_id,
        patient_uid=patient_uid,
        sex=sex,
        birth_date=birth_date,
        eye_sides_by_study=eye_sides_by_study,
        scan_pattern_by_series=scan_pattern_by_series,
        source_paths=source_paths,
    )


def _resolve_laterality_per_series(
    pc: PatientContext,
    bundles: dict[int, SeriesBundle],
) -> None:
    """Fill pc.laterality_by_series.

    Strategy (mirrors assemble_hdf5.py — HEYEX convention "OD scanned first"):
      * For each studyID, look up eye_sides_by_study[study].
      * If only one side present: every series in that study gets that side.
      * If both OD and OS present:
          - sort that study's series by first-B-scan acquisition time;
          - take the two macular-cube candidates (>= 15 B-scans) as anchors;
          - each series picks whichever anchor (earlier=OD, later=OS) is
            closer in time. Series without B-scan meta are skipped.
      * If neither: leave undefined (caller will hard-fail
        'undetermined_laterality').
    """
    # Group bundles by study + collect each series's first acquisition time.
    bundles_by_study: dict[int, list[SeriesBundle]] = {}
    first_time: dict[int, int] = {}
    for sid, b in bundles.items():
        bundles_by_study.setdefault(b.study_id, []).append(b)
        if b.bscan_metas:
            t = min(m.acquisition_time for (_, m) in b.bscan_metas.values())
            first_time[sid] = t

    for study_id, study_bundles in bundles_by_study.items():
        sides = pc.eye_sides_by_study.get(study_id, set())

        if len(sides) == 1:
            only = next(iter(sides))
            for b in study_bundles:
                pc.laterality_by_series[b.series_id] = only
            continue

        if sides == {"OD", "OS"}:
            # Sort all series in the study by acquisition time.
            timed = [
                (first_time[b.series_id], b.series_id)
                for b in study_bundles
                if b.series_id in first_time
            ]
            timed.sort()
            # Anchor on macular-volume-sized series (>= 15 B-scans).
            volume_sids = [
                sid for _, sid in timed
                if len(bundles[sid].bscan_metas) >= 15
            ]
            if len(volume_sids) >= 2:
                anchor_od_t = first_time[volume_sids[0]]
                anchor_os_t = first_time[volume_sids[1]]
                for t, sid in timed:
                    pc.laterality_by_series[sid] = (
                        "OD" if abs(t - anchor_od_t) <= abs(t - anchor_os_t)
                        else "OS"
                    )
            else:
                # No clear pair of volume anchors — fall back to midpoint split.
                ordered_sids = [sid for _, sid in timed]
                mid = len(ordered_sids) // 2
                for sid in ordered_sids[:mid]:
                    pc.laterality_by_series[sid] = "OD"
                for sid in ordered_sids[mid:]:
                    pc.laterality_by_series[sid] = "OS"


def _collect_series(
    entries_with_files: list[tuple[dict, str]],
) -> dict[int, SeriesBundle]:
    """Group OCT-relevant entries by seriesID."""
    bundles: dict[int, SeriesBundle] = {}
    file_cache: dict[str, Any] = {}
    try:
        for entry, filepath in entries_with_files:
            sid = entry.get("seriesID", -1)
            if sid < 0:
                continue
            t = entry["type"]
            if t not in (TYPE_IMAGE, TYPE_OCTA, TYPE_BSCAN_META, TYPE_SEG):
                continue

            if filepath not in file_cache:
                file_cache[filepath] = open(filepath, "rb")
            f = file_cache[filepath]

            bundle = bundles.setdefault(
                sid, SeriesBundle(series_id=sid, study_id=entry.get("studyID", -1))
            )

            if t == TYPE_OCTA:
                bundle.oct_type_hex = TYPE_OCTA
                continue  # OCTA filtered out later

            if t == TYPE_IMAGE:
                sub = entry.get("subID", -1)
                if sub == 0:
                    bundle.slo_image = entry
                elif sub == 1:
                    bundle.oct_images[entry["imageID"]] = entry
            elif t == TYPE_BSCAN_META:
                meta = _read_bscan_meta(f, entry)
                if meta is not None:
                    bundle.bscan_metas[entry["imageID"]] = (entry, meta)
            elif t == TYPE_SEG:
                img_id = entry["imageID"]
                seg_type = struct.unpack_from(
                    "<I", _peek_seg_header(f, entry), 8
                )[0]
                bundle.seg_by_image.setdefault(img_id, {})[int(seg_type)] = entry
    finally:
        for f in file_cache.values():
            f.close()

    return bundles


def _peek_seg_header(f, entry: dict) -> bytes:
    f.seek(data_content_offset(entry))
    return f.read(L.SEG_HEADER_SIZE)


# =========================================================================
# Sample building
# =========================================================================


def _resolve_source_paths(pc: PatientContext) -> tuple[str, str, str]:
    return (
        pc.source_paths.get(".sdb", ""),
        pc.source_paths.get(".edb", ""),
        pc.source_paths.get(".pdb", ""),
    )


def build_sample_for_series(
    bundle: SeriesBundle,
    pc: PatientContext,
    entries_with_files: list[tuple[dict, str]],
) -> tuple[HDF5Sample | None, str | None, MacularCubeVerdict | None]:
    """Attempt to build a sample from a single series bundle.

    Returns (sample_or_None, hard_fail_reason_or_None, verdict).
    """
    # Ordered B-scans (D-axis = superior -> inferior by posY1 desc).
    all_metas = [m for _, m in bundle.bscan_metas.values()]
    if not all_metas:
        return None, "read_series:no_bscan_meta", None
    ordered_metas = C.sort_bscans_superior_to_inferior(all_metas)

    # Image shape from the first real OCT image entry (not metadata)
    first_bscan_image = None
    file_cache: dict[str, Any] = {}
    try:
        def _open(fp: str):
            if fp not in file_cache:
                file_cache[fp] = open(fp, "rb")
            return file_cache[fp]

        # Collect all OCT image entries in D-axis order, paired with filepaths.
        # Use imageID -> filepath map from entries_with_files.
        filepath_by_image: dict[int, str] = {}
        for ent, fp in entries_with_files:
            if (
                ent["type"] == TYPE_IMAGE
                and ent.get("subID", -1) == 1
                and ent.get("seriesID", -1) == bundle.series_id
            ):
                filepath_by_image[ent["imageID"]] = fp

        ordered_oct_entries: list[tuple[dict, str]] = []
        ordered_akt: list[C.BScanMeta] = []
        for meta_entry, meta in [
            (bundle.bscan_metas[img_id][0], m)
            for (img_id, (_, m)) in sorted(
                bundle.bscan_metas.items(),
                key=lambda kv: (-float(kv[1][1].pos_y1), int(kv[1][1].akt_image)),
            )
        ]:
            img_id = meta_entry["imageID"]
            oct_entry = bundle.oct_images.get(img_id)
            if oct_entry is None:
                continue
            fp = filepath_by_image.get(img_id)
            if fp is None:
                continue
            ordered_oct_entries.append((oct_entry, fp))
            ordered_akt.append(meta)

        if not ordered_oct_entries:
            return None, "read_series:no_oct_images", None

        # Peek first image header for shape.
        first_entry, first_fp = ordered_oct_entries[0]
        first_hdr = _read_image_header(_open(first_fp), first_entry)
        if first_hdr is None:
            return None, "read_series:bad_image_header", None

        bscan_h = first_hdr["breite"]
        bscan_w = first_hdr["hoehe"]

        # Macular cube check
        verdict = is_macular_cube(
            bscans=ordered_akt,
            type_hex=bundle.oct_type_hex,
            bscan_h=bscan_h,
            bscan_w=bscan_w,
        )
        if not verdict:
            return None, f"not_macular_volume:{verdict.reason}", verdict

        # Laterality (pre-resolved per-series; honours HEYEX OD-first ordering
        # when a study has both eyes' EyeData entries at study level).
        lat = pc.laterality_by_series.get(bundle.series_id)
        if lat is None:
            return None, "undetermined_laterality", verdict

        # Read volume pixels
        d = len(ordered_oct_entries)
        volume = np.zeros((d, bscan_h, bscan_w), dtype=np.float32)
        for i, (oct_entry, fp) in enumerate(ordered_oct_entries):
            f = _open(fp)
            hdr = _read_image_header(f, oct_entry)
            if hdr is None:
                return None, "corrupt_pixel_data:header", verdict
            if hdr["breite"] != bscan_h or hdr["hoehe"] != bscan_w:
                return None, "corrupt_pixel_data:shape_mismatch", verdict
            pixels = _read_image_pixels(f, oct_entry, hdr)
            if pixels is None:
                return None, "corrupt_pixel_data:short", verdict
            if pixels.dtype == np.uint8:
                # Shouldn't happen for OCT; zero-extend to uint16.
                pixels = pixels.astype(np.uint16)
            volume[i] = pixels

        # Read IR (SLO) image if present
        if bundle.slo_image is not None:
            slo_entry = bundle.slo_image
            slo_fp = None
            for ent, fp in entries_with_files:
                if ent is slo_entry:
                    slo_fp = fp
                    break
            if slo_fp is not None:
                slo_f = _open(slo_fp)
                slo_hdr = _read_image_header(slo_f, slo_entry)
                if slo_hdr is not None:
                    pixels = _read_image_pixels(slo_f, slo_entry, slo_hdr)
                    if pixels is not None and pixels.dtype == np.uint8:
                        ir_img = pixels
                        slo_w_px = ir_img.shape[1]
                        slo_h_px = ir_img.shape[0]
                    else:
                        ir_img = np.zeros((0, 0), dtype=np.uint8)
                        slo_w_px = slo_h_px = 0
                else:
                    ir_img = np.zeros((0, 0), dtype=np.uint8)
                    slo_w_px = slo_h_px = 0
            else:
                ir_img = np.zeros((0, 0), dtype=np.uint8)
                slo_w_px = slo_h_px = 0
        else:
            ir_img = np.zeros((0, 0), dtype=np.uint8)
            slo_w_px = slo_h_px = 0
        has_ir = ir_img.size > 0

        # Per-B-scan IR positions (ascan_pos_ir)
        ascan_pos_ir = np.zeros((d, bscan_w, 2), dtype=np.float32)
        if has_ir:
            fov = C.fov_from_slo_width(slo_w_px)
            for i, meta in enumerate(ordered_akt):
                ascan_pos_ir[i] = C.per_ascan_ir_positions(
                    meta.pos_x1, meta.pos_y1, meta.pos_x2, meta.pos_y2,
                    n_ascans=bscan_w, slo_w=slo_w_px, slo_h=slo_h_px, fov=fov,
                )

        # Layers (stack per B-scan in the same D order)
        ilm = np.full((d, bscan_w), np.nan, dtype=np.float32)
        rpe_bm = np.full((d, bscan_w), np.nan, dtype=np.float32)
        bm_true = np.full((d, bscan_w), np.nan, dtype=np.float32)
        seg_types_seen: set[int] = set()
        bm_true_any_valid = False

        for i, meta in enumerate(ordered_akt):
            entry_meta = None
            for k, (e, m) in bundle.bscan_metas.items():
                if m is meta:
                    entry_meta = e
                    break
            if entry_meta is None:
                continue
            img_id = entry_meta["imageID"]
            seg_by_type = bundle.seg_by_image.get(img_id, {})
            chunks_by_type: dict[int, L.SegChunk] = {}
            for seg_type, seg_entry in seg_by_type.items():
                fp_seg = None
                for ent, fp in entries_with_files:
                    if ent is seg_entry:
                        fp_seg = fp
                        break
                if fp_seg is None:
                    continue
                chunk = _read_seg_chunk(_open(fp_seg), seg_entry)
                if chunk is not None:
                    chunks_by_type[int(seg_type)] = chunk
                    seg_types_seen.add(int(seg_type))

            packed = L.pack_layers(chunks_by_type, n_ascans=bscan_w)
            if "ILM" in packed:
                ilm[i] = packed["ILM"]
            if "RPE_BM" in packed:
                rpe_bm[i] = packed["RPE_BM"]
            if "BM_true" in packed:
                bm_true[i] = packed["BM_true"]
                bm_true_any_valid = True

        if not (np.isfinite(ilm).any() and np.isfinite(rpe_bm).any()):
            return None, "missing_layer_segmentation", verdict

        valid_mask = np.isfinite(ilm) & np.isfinite(rpe_bm)
        valid_count = int(valid_mask.sum())
        valid_ratio = float(valid_count / (d * bscan_w)) if (d * bscan_w) else 0.0

        # H-axis check
        h_axis_fail, _ = check_h_axis(ilm, rpe_bm)
        if h_axis_fail is not None:
            return None, h_axis_fail, verdict

        # Scales
        scale_axial_mm_per_px = float(np.median([m.scale_y for m in ordered_akt]))
        scale_axial_um_per_px = scale_axial_mm_per_px * 1000.0
        # lateral: B-scan length mm / A-scan count
        first_len_mm = (
            ((ordered_akt[0].pos_x2 - ordered_akt[0].pos_x1) ** 2
             + (ordered_akt[0].pos_y2 - ordered_akt[0].pos_y1) ** 2) ** 0.5
        ) * MM_PER_DEG
        scale_lateral_mm_per_px = (
            first_len_mm / bscan_w if bscan_w else 0.0
        )

        # B-scan spacing (deg per index, mm spacing)
        y_degs = np.array([m.pos_y1 for m in ordered_akt], dtype=np.float64)
        if len(y_degs) >= 2:
            spacing_deg = float(np.median(np.abs(np.diff(y_degs))))
        else:
            spacing_deg = 0.0
        scale_bscan_spacing_mm = spacing_deg * MM_PER_DEG

        # Image quality: median across B-scans
        q_per_bscan = np.array(
            [m.image_quality for m in ordered_akt], dtype=np.float32
        )
        median_quality = float(np.median(q_per_bscan)) if q_per_bscan.size else 0.0

        # Thickness median (for extreme_thickness flag)
        thickness = np.where(valid_mask, rpe_bm - ilm, np.nan)
        median_thickness = float(np.nanmedian(thickness)) if np.isfinite(
            thickness
        ).any() else float("nan")

        # Fovea
        fx, fy = estimate_fovea_ir_xy(
            ilm_y=ilm,
            rpe_bm_y=rpe_bm,
            valid_mask=valid_mask,
            ascan_pos_ir=ascan_pos_ir,
            n_bscans=d,
        )
        fovea_valid = bool(np.isfinite(fx) and np.isfinite(fy))

        # Soft flags
        soft_flags = compute_soft_flags(
            valid_ascan_ratio=valid_ratio,
            image_quality=median_quality,
            bscan_spacing_deg=y_degs,
            median_thickness_px=median_thickness,
            has_ir=has_ir,
            fovea_valid=fovea_valid,
        )

        # Acquisition time (use first B-scan's time as series time)
        acq_time_utc = filetime_to_utc_iso(ordered_akt[0].acquisition_time) or ""
        visit_id = iso_to_visit_id(acq_time_utc) or (
            f"series{bundle.series_id}"
        )

        # Age at visit: needs both birth_date (YYYY-MM-DD) and visit date.
        age_at_visit = float("nan")
        if pc.birth_date and acq_time_utc:
            try:
                bd = datetime.strptime(pc.birth_date, "%Y-%m-%d")
                vd = datetime.fromisoformat(acq_time_utc).replace(tzinfo=None)
                age_at_visit = (vd - bd).days / 365.25
            except (ValueError, TypeError):
                age_at_visit = float("nan")

        seg_types_available = L.segmentation_types_available(seg_types_seen)

        sdb, edb, pdb = _resolve_source_paths(pc)

        sample = HDF5Sample(
            patient_id=pc.patient_id,
            visit_date=acq_time_utc,
            visit_id=visit_id,
            acquisition_time_utc=acq_time_utc,
            laterality=lat,
            sex=pc.sex,
            birth_date=pc.birth_date,
            age_at_visit_years=age_at_visit,
            volume=volume,
            ir=ir_img,
            ilm_y=ilm,
            rpe_bm_y=rpe_bm,
            bm_true_y=bm_true if bm_true_any_valid else None,
            valid_ascan_mask=valid_mask,
            ascan_pos_ir=ascan_pos_ir
            if has_ir
            else np.zeros((0, 0, 0), dtype=np.float32),
            image_quality_per_bscan=q_per_bscan,
            line_scans=[],
            scale_axial_um_per_px=scale_axial_um_per_px,
            scale_lateral_mm_per_px=scale_lateral_mm_per_px,
            scale_bscan_spacing_mm=scale_bscan_spacing_mm,
            bscan_spacing_deg_per_index=spacing_deg,
            image_quality=median_quality,
            valid_ascan_count=valid_count,
            valid_ascan_ratio=valid_ratio,
            segmentation_types_available=seg_types_available,
            has_ir=has_ir,
            has_line_scans=False,
            n_line_scans=0,
            fovea_ir_x=fx,
            fovea_ir_y=fy,
            source_sdb_path=sdb,
            source_edb_path=edb,
            source_pdb_path=pdb,
            flags=",".join(soft_flags),
        )
        return sample, None, verdict
    finally:
        for f in file_cache.values():
            f.close()


# =========================================================================
# Top-level entry
# =========================================================================


@dataclass
class ExtractOutcome:
    samples: list[HDF5Sample]
    hard_failures: list[dict]  # for failure_log


def extract_samples_from_pat(pat_dir: str | os.PathLike) -> ExtractOutcome:
    """Extract one or more macular-cube samples from a .pat directory.

    Never raises on soft issues; it records them as `hard_failures` entries.
    """
    pat_dir = str(pat_dir)
    entries_with_files = scan_all_files(pat_dir)

    if not entries_with_files:
        return ExtractOutcome(
            samples=[],
            hard_failures=[{
                "failure_stage": "open_sdb",
                "reason": "no_entries_found",
                "source_sdb_path": pat_dir,
            }],
        )

    pc = _collect_patient(entries_with_files)
    sdb, edb, pdb = _resolve_source_paths(pc)
    bundles = _collect_series(entries_with_files)
    _resolve_laterality_per_series(pc, bundles)

    samples: list[HDF5Sample] = []
    hard_failures: list[dict] = []

    for sid, bundle in sorted(bundles.items()):
        try:
            sample, reason, verdict = build_sample_for_series(
                bundle, pc, entries_with_files
            )
        except Exception as exc:  # noqa: BLE001
            hard_failures.append({
                "failure_stage": "read_series",
                "reason": "exception",
                "source_sdb_path": sdb,
                "source_edb_path": edb,
                "study_id": bundle.study_id,
                "series_id": sid,
                "patient_id": pc.patient_id,
                "exception": repr(exc),
            })
            continue

        if sample is None:
            if reason and reason.startswith("not_macular_volume"):
                continue
            # Metadata-only series (BScanMeta exists but no pixel data)
            # are orphan auxiliary entries, not failures.
            if reason in ("read_series:no_oct_images",
                          "read_series:no_bscan_meta"):
                continue
            hard_failures.append({
                "failure_stage": "validate",
                "reason": reason or "unknown",
                "source_sdb_path": sdb,
                "source_edb_path": edb,
                "study_id": bundle.study_id,
                "series_id": sid,
                "patient_id": pc.patient_id,
            })
            continue

        samples.append(sample)

    selected = _pick_best_per_eye_visit(samples)
    return ExtractOutcome(samples=selected, hard_failures=hard_failures)


def _pick_best_per_eye_visit(samples: list[HDF5Sample]) -> list[HDF5Sample]:
    """Group by (patient_id, laterality, visit_date) and keep the best."""
    by_key: dict[tuple[str, str, str], list[tuple[tuple, HDF5Sample]]] = {}
    for s in samples:
        key = (s.patient_id, s.laterality, s.visit_date)
        sort_key = eye_visit_sort_key(
            s.image_quality, s.valid_ascan_ratio, s.acquisition_time_utc,
            series_id=0,  # not retained at sample level; tie-break on timestamp+id lost here
        )
        by_key.setdefault(key, []).append((sort_key, s))

    result: list[HDF5Sample] = []
    for key, group in by_key.items():
        group.sort(key=lambda p: p[0])
        result.append(group[0][1])
    return result
