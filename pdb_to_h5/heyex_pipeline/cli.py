"""`python -m heyex_pipeline` entry point (CLAUDE_TASK.md E.2).

Batch-process .pat directories under --input to HDF5 files under --output.
Idempotency: a .pat whose target HDF5 already exists with a matching parser
major version is skipped.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from heyex_pipeline.failure_log import FailureLog
from heyex_pipeline.hdf5_writer import (
    HDF5Sample,
    existing_major_matches,
    h5_relative_path,
    write_sample,
)
from heyex_pipeline.manifest import ManifestWriter
from heyex_pipeline.sample_extractor import extract_samples_from_pat
from heyex_pipeline.version import parser_major_version, parser_version

logger = logging.getLogger("heyex_pipeline")


def _iter_pat_dirs(input_root: Path):
    """Recursively yield .pat directories under input_root."""
    for p in input_root.rglob("*.pat"):
        if p.is_dir():
            yield p


def _sample_to_manifest_row(s: HDF5Sample, h5_rel: Path) -> dict:
    year = ""
    try:
        if s.acquisition_time_utc:
            year = s.acquisition_time_utc[:4]
    except Exception:
        year = ""
    return {
        "h5_path": str(h5_rel).replace("\\", "/"),
        "patient_id": s.patient_id,
        "visit_date": s.visit_date,
        "visit_id": s.visit_id,
        "visit_uid": f"{s.patient_id}::{s.visit_id}",
        "longitudinal_key": f"{s.patient_id}::{s.laterality}",
        "laterality": s.laterality,
        "acquisition_time_utc": s.acquisition_time_utc,
        "acquisition_year": year,
        "image_quality": float(s.image_quality),
        "valid_ascan_ratio": float(s.valid_ascan_ratio),
        "n_bscans": int(s.volume.shape[0]),
        "bscan_height": int(s.volume.shape[1]),
        "bscan_width": int(s.volume.shape[2]),
        "scale_axial_um_per_px": float(s.scale_axial_um_per_px),
        "scale_lateral_mm_per_px": float(s.scale_lateral_mm_per_px),
        "scale_bscan_spacing_mm": float(s.scale_bscan_spacing_mm),
        "has_ir": bool(s.has_ir),
        "has_line_scans": bool(s.has_line_scans),
        "n_line_scans": int(s.n_line_scans),
        "segmentation_types_available": s.segmentation_types_available,
        "has_bm_true": s.bm_true_y is not None,
        "fovea_valid": bool(
            s.fovea_ir_x == s.fovea_ir_x and s.fovea_ir_y == s.fovea_ir_y
        ),  # NaN-safe: NaN != NaN
        "flags": s.flags,
        "parser_version": parser_version,
        "sex": s.sex,
        "birth_date": s.birth_date,
        "age_at_visit_years": float(s.age_at_visit_years),
    }


def _process_one_pat(pat_dir: str, output_root: str, dry_run: bool) -> dict:
    """Worker entry point. Returns a dict summarising what happened."""
    result = {
        "pat_dir": pat_dir,
        "wrote": [],       # list of (rel_path, manifest_row)
        "skipped": [],     # list of (rel_path, reason)
        "failures": [],    # list of dicts for failure_log
    }
    try:
        outcome = extract_samples_from_pat(pat_dir)
    except Exception as exc:  # noqa: BLE001
        result["failures"].append({
            "failure_stage": "open_sdb",
            "reason": "exception_in_extract",
            "source_sdb_path": pat_dir,
            "exception": f"{exc!r}\n{traceback.format_exc()}",
        })
        return result

    result["failures"].extend(outcome.hard_failures)
    output_root_p = Path(output_root)

    for sample in outcome.samples:
        rel = h5_relative_path(sample.patient_id, sample.visit_id, sample.laterality)
        target = output_root_p / rel

        if existing_major_matches(target, parser_major_version()):
            result["skipped"].append((str(rel), "existing_major_matches"))
            continue

        if dry_run:
            result["wrote"].append((str(rel), _sample_to_manifest_row(sample, rel)))
            continue

        try:
            write_sample(sample, output_root_p)
            result["wrote"].append((str(rel), _sample_to_manifest_row(sample, rel)))
        except Exception as exc:  # noqa: BLE001
            result["failures"].append({
                "failure_stage": "write",
                "reason": "write_exception",
                "source_sdb_path": pat_dir,
                "patient_id": sample.patient_id,
                "series_id": "",
                "exception": repr(exc),
            })

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m heyex_pipeline",
        description="Convert Heidelberg .pat directories to training HDF5.",
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Root dir to recursively search for .pat folders")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output root for HDF5 + manifest + failures.jsonl")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes")
    parser.add_argument("--manifest-checkpoint-interval", type=int, default=10_000,
                        dest="ckpt_interval")
    parser.add_argument("--dry-run", action="store_true",
                        help="Enumerate what would be written; do not touch disk")
    parser.add_argument("--verify-samples", type=int, default=0,
                        help="After run, verify N random .h5 (Stage 5 tool)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if not args.input.exists():
        logger.error("--input %s does not exist", args.input)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    failure_log = FailureLog(args.output / "failures.jsonl")
    manifest = ManifestWriter(
        args.output / "manifest.parquet",
        checkpoint_interval=args.ckpt_interval,
    )

    pat_dirs = list(_iter_pat_dirs(args.input))
    logger.info("Found %d .pat directories under %s", len(pat_dirs), args.input)

    total_written = 0
    total_skipped = 0
    total_failed = 0

    if args.workers <= 1:
        for pd in pat_dirs:
            res = _process_one_pat(str(pd), str(args.output), args.dry_run)
            total_written += _consume(res, manifest, failure_log)[0]
            total_skipped += len(res["skipped"])
            total_failed += len(res["failures"])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(_process_one_pat, str(pd), str(args.output), args.dry_run): pd
                for pd in pat_dirs
            }
            for fut in as_completed(futs):
                res = fut.result()
                total_written += _consume(res, manifest, failure_log)[0]
                total_skipped += len(res["skipped"])
                total_failed += len(res["failures"])

    manifest.close()

    logger.info(
        "Done. written=%d skipped=%d failed=%d", total_written, total_skipped, total_failed
    )
    sys.stdout.write(
        f"written={total_written} skipped={total_skipped} failed={total_failed}\n"
    )

    if args.verify_samples > 0 and total_written > 0:
        _verify_random(args.output, args.verify_samples)

    return 0


def _consume(res: dict, manifest: ManifestWriter, failure_log: FailureLog) -> tuple[int, int]:
    written = 0
    for _, row in res["wrote"]:
        manifest.add_row(row)
        written += 1
    for fail in res["failures"]:
        failure_log.log(**fail)
    return written, len(res["failures"])


def _verify_random(output_root: Path, n: int) -> None:
    import random
    h5_files = [p for p in output_root.rglob("*.h5")]
    if not h5_files:
        return
    chosen = random.sample(h5_files, k=min(n, len(h5_files)))
    from tools.verify_sample import verify_sample  # lazy import
    verify_root = output_root / "_verify"
    verify_root.mkdir(exist_ok=True)
    for p in chosen:
        out = verify_root / (p.stem + ".png")
        try:
            verify_sample(p, out)
        except Exception as exc:  # noqa: BLE001
            logger.warning("verify_sample failed for %s: %r", p, exc)


if __name__ == "__main__":
    raise SystemExit(main())
