"""
Export IR Localizer mapping: each OCT B-scan's position on the IR/SLO image.

For each B-scan, outputs:
- The B-scan line endpoints in degree coordinates
- The corresponding SLO/IR pixel coordinates
- SLO image metadata (size, FOV)

Mapping formula:
  SLO_pixel_X = (x_deg / FOV + 0.5) * SLO_width
  SLO_pixel_Y = (0.5 - y_deg / FOV) * SLO_height

Where FOV is determined by SLO image resolution:
  768x768 -> 30° FOV
  384x384 -> 15° FOV
"""
import struct, os, sys, csv, math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from export_e2e_csv import (scan_all_files, data_content_offset,
                             DATA_ENTRY_HEADER_SIZE, windows_filetime_to_str)

# SLO FOV lookup by resolution
SLO_FOV_MAP = {
    768: 30.0,  # 768x768 -> 30 degree FOV
    384: 15.0,  # 384x384 -> 15 degree FOV
}

# Standard Gullstrand eye model: 1 degree ≈ 0.288 mm on retina
# Can be corrected with Littmann-Bennett formula using axial length:
#   q = 0.01306 * (AL_mm - 1.82); corrected_mm = q * deg * (pi/180) * 1000
# For average eye (AL=23.9mm): q ≈ 0.2882 → very close to 0.288
MM_PER_DEG = 0.288

BSCAN_META_SIZE = 0xA0


def deg_to_slo_pixel(x_deg, y_deg, fov, slo_w, slo_h):
    """Convert degree coordinates to SLO pixel coordinates."""
    if fov == 0:
        return None, None
    px = (x_deg / fov + 0.5) * slo_w
    py = (0.5 - y_deg / fov) * slo_h
    return round(px, 2), round(py, 2)


def export_ir_localizer(pat_dir, output_dir):
    """Export IR Localizer CSVs for a single .pat directory."""
    os.makedirs(output_dir, exist_ok=True)
    all_entries = scan_all_files(pat_dir)

    # 1. Get SLO image dimensions per series
    slo_info = {}
    for entry, filepath in all_entries:
        if entry['type'] != 0x40000000 or entry['subID'] != 0:
            continue
        sid = entry['seriesID']
        with open(filepath, 'rb') as f:
            f.seek(entry['dataAddress'] + DATA_ENTRY_HEADER_SIZE)
            hdr = f.read(20)
            if len(hdr) < 20:
                continue
            img_type = hdr[6]
            width = struct.unpack_from('<I', hdr, 12)[0]
            height = struct.unpack_from('<I', hdr, 16)[0]
        fov = SLO_FOV_MAP.get(width, 0)
        slo_info[sid] = {
            'width': width, 'height': height, 'fov': fov,
            'imgType': img_type, 'dataLen': entry['dataLength'],
            'file': entry['_sourceFile'],
        }

    # 2. Get BScan metadata (positions and scan info)
    bscan_meta = {}
    for entry, filepath in all_entries:
        if entry['type'] != 0x2714:
            continue
        sid = entry['seriesID']
        iid = entry['imageID']
        with open(filepath, 'rb') as f:
            f.seek(data_content_offset(entry))
            raw = f.read(min(entry['dataLength'], BSCAN_META_SIZE))
            if len(raw) < 0x60:
                continue
            imgSizeX = struct.unpack_from('<I', raw, 4)[0]
            imgSizeY = struct.unpack_from('<I', raw, 8)[0]
            posX1 = struct.unpack_from('<f', raw, 12)[0]
            posY1 = struct.unpack_from('<f', raw, 16)[0]
            posX2 = struct.unpack_from('<f', raw, 20)[0]
            posY2 = struct.unpack_from('<f', raw, 24)[0]
            scaleY = struct.unpack_from('<f', raw, 36)[0]
            numImages = struct.unpack_from('<I', raw, 0x44)[0]
            aktImage = struct.unpack_from('<I', raw, 0x48)[0]
            scanType = struct.unpack_from('<I', raw, 0x4c)[0]
            centerX = struct.unpack_from('<f', raw, 0x50)[0]
            centerY = struct.unpack_from('<f', raw, 0x54)[0]
            acqTime = struct.unpack_from('<Q', raw, 0x5c)[0]
            imageQuality = struct.unpack_from('<f', raw, 0x9c)[0] if len(raw) >= 0xa0 else 0

        bscan_meta[(sid, iid)] = {
            'imgSizeX': imgSizeX, 'imgSizeY': imgSizeY,
            'posX1': posX1, 'posY1': posY1, 'posX2': posX2, 'posY2': posY2,
            'scaleY': scaleY, 'numImages': numImages, 'aktImage': aktImage,
            'scanType': scanType, 'centerX': centerX, 'centerY': centerY,
            'acqTime': acqTime, 'imageQuality': imageQuality,
            'sourceFile': entry['_sourceFile'],
        }

    # 3. Export CSVs
    # --- CSV 15: IR Localizer Summary (per series) ---
    csv15_path = os.path.join(output_dir, '15_ir_localizer_summary.csv')
    with open(csv15_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seriesID', 'SLO_width', 'SLO_height', 'SLO_FOV_deg',
            'SLO_deg_per_pixel', 'SLO_um_per_pixel', 'SLO_FOV_mm',
            'SLO_imgType', 'SLO_dataSizeBytes',
            'scanCenter_X_deg', 'scanCenter_Y_deg',
            'scanCenter_X_pixel', 'scanCenter_Y_pixel',
            'numBScans', 'scanType', 'sourceFile'
        ])
        for sid in sorted(slo_info.keys()):
            si = slo_info[sid]
            first_meta = None
            for (s, iid), m in sorted(bscan_meta.items()):
                if s == sid:
                    first_meta = m
                    break
            cx_deg = first_meta['centerX'] if first_meta else 0
            cy_deg = first_meta['centerY'] if first_meta else 0
            num_bscan = sum(1 for (s, _) in bscan_meta if s == sid)
            scan_type = first_meta['scanType'] if first_meta else 0
            cx_px, cy_px = deg_to_slo_pixel(cx_deg, cy_deg, si['fov'], si['width'], si['height'])
            deg_per_px = round(si['fov'] / si['width'], 6) if si['width'] > 0 else 0
            um_per_px = round(deg_per_px * MM_PER_DEG * 1000, 2) if deg_per_px else ''
            fov_mm = round(si['fov'] * MM_PER_DEG, 2)
            writer.writerow([
                sid, si['width'], si['height'], si['fov'],
                deg_per_px, um_per_px, fov_mm,
                si['imgType'], si['dataLen'],
                cx_deg, cy_deg, cx_px, cy_px,
                num_bscan, scan_type, si['file']
            ])
    print(f"  15_ir_localizer_summary.csv: {len(slo_info)} rows written")

    # --- CSV 16: IR Localizer Detail (per B-scan) ---
    csv16_path = os.path.join(output_dir, '16_ir_localizer_bscan.csv')
    rows_written = 0
    with open(csv16_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seriesID', 'imageID', 'BScan_index',
            'posX1_deg', 'posY1_deg', 'posX2_deg', 'posY2_deg',
            'BScan_length_deg', 'BScan_length_mm',
            'SLO_x1_pixel', 'SLO_y1_pixel', 'SLO_x2_pixel', 'SLO_y2_pixel',
            'SLO_width', 'SLO_height', 'SLO_FOV_deg',
            'BScan_imgSizeX', 'BScan_imgSizeY',
            'centerX_deg', 'centerY_deg',
            'acquisitionTime', 'imageQuality',
            'sourceFile'
        ])
        for (sid, iid) in sorted(bscan_meta.keys()):
            m = bscan_meta[(sid, iid)]
            si = slo_info.get(sid)
            if not si:
                continue
            bscan_idx = iid // 2
            x1_px, y1_px = deg_to_slo_pixel(m['posX1'], m['posY1'],
                                              si['fov'], si['width'], si['height'])
            x2_px, y2_px = deg_to_slo_pixel(m['posX2'], m['posY2'],
                                              si['fov'], si['width'], si['height'])
            acq_str = windows_filetime_to_str(m['acqTime'])
            line_deg = math.sqrt((m['posX2']-m['posX1'])**2 + (m['posY2']-m['posY1'])**2)
            line_mm = round(line_deg * MM_PER_DEG, 4)
            writer.writerow([
                sid, iid, bscan_idx,
                round(m['posX1'], 4), round(m['posY1'], 4),
                round(m['posX2'], 4), round(m['posY2'], 4),
                round(line_deg, 4), line_mm,
                x1_px, y1_px, x2_px, y2_px,
                si['width'], si['height'], si['fov'],
                m['imgSizeX'], m['imgSizeY'],
                round(m['centerX'], 4), round(m['centerY'], 4),
                acq_str, round(m['imageQuality'], 2),
                m['sourceFile']
            ])
            rows_written += 1
    print(f"  16_ir_localizer_bscan.csv: {rows_written} rows written")

    # --- CSV 17: Per-Ascan SLO pixel mapping ---
    csv17_path = os.path.join(output_dir, '17_ir_localizer_ascan.csv')
    ascan_rows = 0
    with open(csv17_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seriesID', 'imageID', 'BScan_index', 'ascanIndex',
            'SLO_x_pixel', 'SLO_y_pixel',
            'x_deg', 'y_deg'
        ])
        for (sid, iid) in sorted(bscan_meta.keys()):
            m = bscan_meta[(sid, iid)]
            si = slo_info.get(sid)
            if not si:
                continue
            bscan_idx = iid // 2
            num_ascans = m['imgSizeX']
            for a in range(num_ascans):
                t = a / (num_ascans - 1) if num_ascans > 1 else 0.5
                x_deg = m['posX1'] + t * (m['posX2'] - m['posX1'])
                y_deg = m['posY1'] + t * (m['posY2'] - m['posY1'])
                slo_x, slo_y = deg_to_slo_pixel(x_deg, y_deg,
                                                  si['fov'], si['width'], si['height'])
                writer.writerow([
                    sid, iid, bscan_idx, a,
                    slo_x, slo_y,
                    round(x_deg, 4), round(y_deg, 4)
                ])
                ascan_rows += 1
    print(f"  17_ir_localizer_ascan.csv: {ascan_rows} rows written")
    print(f"\nDone! IR Localizer CSVs exported to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export IR Localizer mapping')
    parser.add_argument('pat_dirs', nargs='*',
                        help='.pat directories to process (default: ./00000004.pat)')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output base directory')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    if not args.pat_dirs:
        pat_dir = os.path.join(base, '00000004.pat')
        output_dir = args.output_dir or os.path.join(base, 'csv_output')
        print(f"Source: {pat_dir}")
        export_ir_localizer(pat_dir, output_dir)
    else:
        for pd in args.pat_dirs:
            pd = os.path.abspath(pd)
            if not os.path.isdir(pd):
                print(f"Warning: {pd} not found, skipping")
                continue
            pat_name = os.path.basename(pd).replace('.pat', '')
            if args.output_dir:
                out = os.path.join(args.output_dir, pat_name)
            else:
                out = os.path.join(base, 'batch_output', pat_name)
            print(f"\n{'='*60}")
            print(f"Processing IR Localizer: {pd}")
            print(f"{'='*60}")
            export_ir_localizer(pd, out)


if __name__ == '__main__':
    main()
