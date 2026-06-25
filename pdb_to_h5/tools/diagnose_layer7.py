"""Statistics on segType 7 (true BM) validity across a batch of `.pat` dirs.

For every `.pat` under --input (recursively), scan every SEG chunk with
seg_type == 7 and count how many float values are finite (valid) vs.
FLT_MAX sentinels. Output one CSV row per series plus a summary line.

Usage:
    python tools/diagnose_layer7.py --input /data/raw --output layer7.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "existing") not in sys.path:
    sys.path.insert(0, str(_ROOT / "existing"))

from export_e2e_csv import data_content_offset, scan_all_files  # noqa: E402
from heyex_pipeline import layers as L  # noqa: E402

TYPE_SEG = 0x2723


def scan_pat(pat_dir: Path) -> list[dict]:
    """Return per-series segType-7 validity rows."""
    entries = scan_all_files(str(pat_dir))
    by_series: dict[int, dict] = {}
    file_handles: dict[str, object] = {}
    try:
        for entry, filepath in entries:
            if entry["type"] != TYPE_SEG:
                continue
            if filepath not in file_handles:
                file_handles[filepath] = open(filepath, "rb")
            f = file_handles[filepath]
            f.seek(data_content_offset(entry))
            payload = f.read(entry["dataLength"])
            chunk = L.parse_seg_chunk(payload)
            if chunk is None or chunk.seg_type != 7:
                continue
            sid = entry.get("seriesID", -1)
            row = by_series.setdefault(sid, {
                "pat_dir": str(pat_dir),
                "series_id": sid,
                "seg7_chunks": 0,
                "seg7_total_ascans": 0,
                "seg7_valid_ascans": 0,
            })
            row["seg7_chunks"] += 1
            row["seg7_total_ascans"] += int(chunk.boundary.size)
            row["seg7_valid_ascans"] += int(np.isfinite(chunk.boundary).sum())
    finally:
        for f in file_handles.values():
            f.close()
    return list(by_series.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit segType-7 validity.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)

    pat_dirs = [p for p in args.input.rglob("*.pat") if p.is_dir()]

    all_rows: list[dict] = []
    for pd in pat_dirs:
        try:
            all_rows.extend(scan_pat(pd))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {pd}: {exc!r}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        cols = ("pat_dir", "series_id", "seg7_chunks",
                "seg7_total_ascans", "seg7_valid_ascans", "valid_ratio")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            total = r["seg7_total_ascans"]
            r["valid_ratio"] = (
                r["seg7_valid_ascans"] / total if total else 0.0
            )
            w.writerow(r)

    total_ascans = sum(r["seg7_total_ascans"] for r in all_rows)
    valid_ascans = sum(r["seg7_valid_ascans"] for r in all_rows)
    ratio = valid_ascans / total_ascans if total_ascans else 0.0
    print(
        f"series={len(all_rows)} "
        f"total_ascans={total_ascans} "
        f"valid_ascans={valid_ascans} "
        f"valid_ratio={ratio:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
