"""JSONL failure log appender (CLAUDE_TASK.md E.5).

One JSON object per hard failure, schema:

    {
      "source_sdb_path": str,
      "source_edb_path": str,
      "study_id": str,
      "series_id": str,
      "patient_id": str,
      "failure_stage": "open_sdb|read_series|parse_layers|validate|write",
      "reason": str,
      "exception": str,
      "parser_version": str,
      "timestamp_utc": str
    }
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from heyex_pipeline.version import parser_version

FAILURE_STAGES = frozenset({
    "open_sdb",
    "read_series",
    "parse_layers",
    "validate",
    "write",
})


class FailureLog:
    """Thread-safe JSONL appender. Safe to construct once per worker."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        *,
        failure_stage: str,
        reason: str,
        source_sdb_path: str = "",
        source_edb_path: str = "",
        study_id: str | int = "",
        series_id: str | int = "",
        patient_id: str = "",
        exception: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        if failure_stage not in FAILURE_STAGES:
            raise ValueError(
                f"failure_stage={failure_stage!r} not in {sorted(FAILURE_STAGES)}"
            )
        record = {
            "source_sdb_path": str(source_sdb_path),
            "source_edb_path": str(source_edb_path),
            "study_id": str(study_id),
            "series_id": str(series_id),
            "patient_id": str(patient_id),
            "failure_stage": failure_stage,
            "reason": reason,
            "exception": exception,
            "parser_version": parser_version,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            record.update(extra)
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
