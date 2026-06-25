#!/usr/bin/env python3
"""
Batch Export: Process multiple .pat directories through all export pipelines.

Usage:
  # Auto-discover all .pat directories under a parent folder:
  python batch_export.py /path/to/parent_folder

  # Explicitly specify .pat directories:
  python batch_export.py /path/to/00000004.pat /path/to/00000005.pat

  # Custom output directory:
  python batch_export.py /path/to/parent_folder -o /path/to/output

  # Skip specific export steps:
  python batch_export.py /path/to/parent_folder --skip-images --skip-slo

Output structure:
  batch_output/
    00000004/
      01_all_entries.csv ... 17_ir_localizer_ascan.csv
      ir_images/
        series_XX_SLO_raw.png
        series_XX_SLO_with_bscans.png
    00000005/
      ...
"""
import os
import sys
import time
import csv
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_e2e_csv import export_csvs
from export_layers import export_layers
from export_ir_localizer import export_ir_localizer
from export_slo_images import export_slo_images
from export_image_details import export_image_details


def merge_csvs(output_base, pat_names):
    """Merge per-folder CSVs into combined CSVs with pat_folder column."""
    merged_dir = os.path.join(output_base, '_merged')
    os.makedirs(merged_dir, exist_ok=True)

    # Discover CSV filenames from first available pat folder
    csv_names = set()
    for pn in pat_names:
        d = os.path.join(output_base, pn)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith('.csv'):
                    csv_names.add(fn)

    csv_names = sorted(csv_names)
    if not csv_names:
        print("  No CSVs found to merge.")
        return

    merged_count = 0
    for csv_name in csv_names:
        all_rows = []
        all_fields = ['pat_folder']
        seen_fields = set(all_fields)

        for pn in pat_names:
            csv_path = os.path.join(output_base, pn, csv_name)
            if not os.path.isfile(csv_path):
                continue
            with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for field in reader.fieldnames:
                        if field not in seen_fields:
                            all_fields.append(field)
                            seen_fields.add(field)
                for row in reader:
                    row['pat_folder'] = pn
                    all_rows.append(row)

        if not all_rows:
            continue

        merged_path = os.path.join(merged_dir, csv_name)
        with open(merged_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction='ignore')
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

        merged_count += 1
        print(f"    {csv_name}: {len(all_rows)} rows from {sum(1 for pn in pat_names if os.path.isfile(os.path.join(output_base, pn, csv_name)))} folders")

    print(f"  Merged {merged_count} CSVs -> {merged_dir}")


def find_pat_dirs(parent):
    """Find all .pat directories under parent (non-recursive)."""
    pat_dirs = []
    if not os.path.isdir(parent):
        return pat_dirs
    for name in sorted(os.listdir(parent)):
        fullpath = os.path.join(parent, name)
        if os.path.isdir(fullpath) and name.endswith('.pat'):
            pat_dirs.append(fullpath)
    return pat_dirs


def process_one(pat_dir, output_dir, skip_images=False, skip_slo=False,
                skip_layers=False, skip_ir=False, skip_image_details=False):
    """Run all export pipelines on a single .pat directory."""
    results = {}
    steps = []

    if not skip_images:
        steps.append(('CSV (11 files)', lambda: export_csvs(pat_dir, output_dir)))
    if not skip_layers:
        steps.append(('Layers (3 files)', lambda: export_layers(pat_dir, output_dir)))
    if not skip_ir:
        steps.append(('IR Localizer (3 files)', lambda: export_ir_localizer(pat_dir, output_dir)))
    if not skip_image_details:
        steps.append(('Image Details (1 file)', lambda: export_image_details(pat_dir, output_dir)))
    if not skip_slo:
        slo_dir = os.path.join(output_dir, 'ir_images')
        steps.append(('SLO Images', lambda: export_slo_images(pat_dir, slo_dir)))

    for name, fn in steps:
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            results[name] = f"OK ({elapsed:.1f}s)"
        except Exception as e:
            elapsed = time.time() - t0
            results[name] = f"FAILED ({elapsed:.1f}s): {e}"
            traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch process multiple .pat directories through all export pipelines.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python batch_export.py .                          # all .pat dirs in current folder
  python batch_export.py /data/heyex               # all .pat dirs under /data/heyex
  python batch_export.py dir1.pat dir2.pat          # specific directories
  python batch_export.py . -o /output --skip-slo    # custom output, skip SLO images
""")
    parser.add_argument('paths', nargs='+',
                        help='Parent directory (auto-discovers .pat subdirs) '
                             'or explicit .pat directories')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output base directory (default: ./batch_output)')
    parser.add_argument('--skip-csv', action='store_true',
                        help='Skip basic CSV export (01-11)')
    parser.add_argument('--skip-layers', action='store_true',
                        help='Skip layer boundary/thickness export (12-14)')
    parser.add_argument('--skip-ir', action='store_true',
                        help='Skip IR Localizer export (15-17)')
    parser.add_argument('--skip-slo', action='store_true',
                        help='Skip SLO image PNG export')
    parser.add_argument('--skip-image-details', action='store_true',
                        help='Skip image details export (18)')
    parser.add_argument('--no-merge', action='store_true',
                        help='Skip merging CSVs into combined files')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    # Resolve .pat directories
    pat_dirs = []
    for p in args.paths:
        p = os.path.abspath(p)
        if os.path.isdir(p) and p.endswith('.pat'):
            pat_dirs.append(p)
        elif os.path.isdir(p):
            found = find_pat_dirs(p)
            if found:
                pat_dirs.extend(found)
            else:
                print(f"Warning: no .pat directories found under {p}")
        else:
            print(f"Warning: {p} does not exist, skipping")

    if not pat_dirs:
        print("Error: No .pat directories found.")
        sys.exit(1)

    output_base = args.output_dir or os.path.join(base, 'batch_output')

    print(f"{'='*60}")
    print(f"Batch Export")
    print(f"{'='*60}")
    print(f"Found {len(pat_dirs)} .pat directories:")
    for d in pat_dirs:
        print(f"  - {d}")
    print(f"Output base: {output_base}")
    print(f"{'='*60}\n")

    total_t0 = time.time()
    all_results = {}
    pat_names_list = []

    for i, pat_dir in enumerate(pat_dirs, 1):
        pat_name = os.path.basename(pat_dir).replace('.pat', '')
        pat_names_list.append(pat_name)
        out_dir = os.path.join(output_base, pat_name)

        print(f"\n{'='*60}")
        print(f"[{i}/{len(pat_dirs)}] {pat_name}")
        print(f"  Source: {pat_dir}")
        print(f"  Output: {out_dir}")
        print(f"{'='*60}")

        results = process_one(
            pat_dir, out_dir,
            skip_images=args.skip_csv,
            skip_slo=args.skip_slo,
            skip_layers=args.skip_layers,
            skip_ir=args.skip_ir,
            skip_image_details=args.skip_image_details,
        )
        all_results[pat_name] = results

    total_elapsed = time.time() - total_t0

    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY  ({total_elapsed:.1f}s total)")
    print(f"{'='*60}")
    for pat_name, results in all_results.items():
        print(f"\n  {pat_name}:")
        for step, status in results.items():
            print(f"    {step}: {status}")

    # Merge CSVs from all folders
    if not args.no_merge and len(pat_names_list) > 0:
        print(f"\n{'='*60}")
        print("Merging CSVs...")
        print(f"{'='*60}")
        merge_csvs(output_base, pat_names_list)

    print(f"\nOutput: {output_base}")
    print("Done!")


if __name__ == '__main__':
    main()
