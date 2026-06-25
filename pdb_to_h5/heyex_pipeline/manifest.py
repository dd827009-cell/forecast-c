"""Parquet manifest (CLAUDE_TASK.md E.4).

Manifest rows are accumulated in memory and flushed to parquet every
`checkpoint_interval` adds (or on `close()`). The flush is atomic
(write to a temp file and rename) so crashes mid-flush do not corrupt
the manifest.

Schema (E.4 row type, maintained explicitly so missing fields default to
a typed null instead of silently dropping):

    h5_path, patient_id, visit_date, visit_id, visit_uid, longitudinal_key,
    laterality, acquisition_time_utc, acquisition_year,
    image_quality, valid_ascan_ratio, n_bscans, bscan_height, bscan_width,
    scale_axial_um_per_px, scale_lateral_mm_per_px, scale_bscan_spacing_mm,
    has_ir, has_line_scans, n_line_scans,
    segmentation_types_available, has_bm_true, fovea_valid,
    flags, parser_version
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

MANIFEST_COLUMNS: tuple[str, ...] = (
    "h5_path",
    "patient_id",
    "visit_date",
    "visit_id",
    "visit_uid",
    "longitudinal_key",
    "laterality",
    "acquisition_time_utc",
    "acquisition_year",
    "image_quality",
    "valid_ascan_ratio",
    "n_bscans",
    "bscan_height",
    "bscan_width",
    "scale_axial_um_per_px",
    "scale_lateral_mm_per_px",
    "scale_bscan_spacing_mm",
    "has_ir",
    "has_line_scans",
    "n_line_scans",
    "segmentation_types_available",
    "has_bm_true",
    "fovea_valid",
    "flags",
    "parser_version",
    "sex",
    "birth_date",
    "age_at_visit_years",
)


class ManifestWriter:
    """Accumulate manifest rows; checkpoint to parquet atomically."""

    def __init__(
        self,
        path: str | os.PathLike,
        checkpoint_interval: int = 10_000,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = int(checkpoint_interval)
        self._rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        # Pre-load any existing manifest so repeated runs accumulate.
        if self.path.exists():
            self._rows = _read_parquet_rows(self.path)

    def add_row(self, row: dict[str, Any]) -> None:
        filtered = {k: row.get(k) for k in MANIFEST_COLUMNS}
        with self._lock:
            self._rows.append(filtered)
            if (
                self.checkpoint_interval > 0
                and len(self._rows) % self.checkpoint_interval == 0
            ):
                self._flush_locked()

    def close(self) -> None:
        with self._lock:
            self._flush_locked()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ----- internals -----

    def _flush_locked(self) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        _write_parquet_rows(tmp_path, self._rows)
        os.replace(tmp_path, self.path)


def _write_parquet_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    df.to_parquet(path, engine="pyarrow", index=False)


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path, engine="pyarrow")
    # Ensure every column exists even if the file is from an earlier version.
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[list(MANIFEST_COLUMNS)].to_dict(orient="records")
