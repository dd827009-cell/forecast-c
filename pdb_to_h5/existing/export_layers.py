#!/usr/bin/env python3
"""
Export retinal layer boundaries, thickness, and derived metrics from E2E segmentation data.
Based on the segmentation data structure from LibE2E.
"""
import struct
import os
import csv
import datetime
from collections import defaultdict

# ---- Reuse parsing from export_e2e_csv.py ----

DIR_ENTRY_FORMAT = '<IIIIiiiihhII'
DIR_ENTRY_SIZE = 44
DATA_ENTRY_HEADER_SIZE = 60  # 16 (DataRawHeader) + 44 (Raw)
SEG_HEADER_FORMAT = '<IIII5I'
SEG_HEADER_SIZE = struct.calcsize(SEG_HEADER_FORMAT)

# Layer type mapping (based on data analysis)
LAYER_NAMES = {
    5: 'ILM',         # Internal Limiting Membrane (真正的 ILM)
    2: 'RPE_BM',      # RPE/BM complex (posterior boundary, typically labeled BM
                      # by commercial software but anatomically represents RPE)
    7: 'BM_true',     # True Bruch's membrane (only available in Advanced RPE
                      # mode; often invalid/missing)
}


def windows_filetime_to_str(filetime):
    try:
        if filetime == 0:
            return ""
        EPOCH_AS_FILETIME = 116444736000000000
        us = (filetime - EPOCH_AS_FILETIME) // 10
        dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=us)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, ValueError, OSError):
        return ""


def sign_extend_16_to_32(v):
    v16 = v & 0xFFFF
    if v16 & 0x8000:
        return 0xFFFF0000 | v16
    return v16


def read_dir_entry(f, addr):
    f.seek(addr)
    data = f.read(DIR_ENTRY_SIZE)
    if len(data) < DIR_ENTRY_SIZE:
        return None
    v = struct.unpack(DIR_ENTRY_FORMAT, data)
    return {
        'indexAddress': v[0], 'dataAddress': v[1], 'dataLength': v[2],
        'zero': v[3], 'patientID': v[4], 'studyID': v[5],
        'seriesID': v[6], 'imageID': v[7], 'subID': v[8],
        'unknown': v[9], 'type': v[10], 'checksum': v[11],
    }


def calc_checksum_dir(e):
    cs = (e['indexAddress'] + e['dataAddress'] + e['dataLength'] + e['zero']
          + (e['patientID'] & 0xFFFFFFFF) + (e['studyID'] & 0xFFFFFFFF)
          + (e['seriesID'] & 0xFFFFFFFF) + (e['imageID'] & 0xFFFFFFFF)
          + sign_extend_16_to_32(e['subID']) + e['type'])
    return (cs - 0x789ABCDF) & 0xFFFFFFFF


def read_mdb_dir(f, link_addr):
    entries = []
    f.seek(link_addr)
    data = f.read(4)
    if len(data) < 4:
        return entries
    act_dir_addr = struct.unpack('<I', data)[0]
    if act_dir_addr == 0:
        return entries
    f.seek(act_dir_addr)
    if f.read(6) != b'MDbDir':
        return entries
    entries.extend(read_mdb_dir(f, act_dir_addr + 0x2c))
    pos = act_dir_addr + 0x34
    while True:
        e = read_dir_entry(f, pos)
        if e is None or e['indexAddress'] != pos:
            break
        if calc_checksum_dir(e) == e['checksum']:
            entries.append(e)
        pos += DIR_ENTRY_SIZE
    return entries


def data_content_offset(entry):
    return entry['dataAddress'] + DATA_ENTRY_HEADER_SIZE


def parse_segmentation(f, entry):
    """Parse segmentation data: SegHeader + float array."""
    f.seek(data_content_offset(entry))
    raw = f.read(entry['dataLength'])
    if len(raw) < SEG_HEADER_SIZE:
        return None
    vals = struct.unpack_from(SEG_HEADER_FORMAT, raw, 0)
    u0, index, seg_type, size = vals[0], vals[1], vals[2], vals[3]
    max_elements = (entry['dataLength'] - SEG_HEADER_SIZE) // 4
    num_elements = min(size, max_elements)
    boundary = []
    for i in range(num_elements):
        offset = SEG_HEADER_SIZE + i * 4
        if offset + 4 > len(raw):
            break
        boundary.append(struct.unpack_from('<f', raw, offset)[0])
    return {
        'segIndex': index,
        'segType': seg_type,
        'numElements': num_elements,
        'boundary': boundary
    }


def parse_bscan_metadata(f, entry):
    """Parse BScan metadata for scaleY."""
    f.seek(data_content_offset(entry))
    raw = f.read(min(entry['dataLength'], 256))
    if len(raw) < 0x60:
        return None
    imgSizeX = struct.unpack_from('<I', raw, 0x04)[0]
    imgSizeY = struct.unpack_from('<I', raw, 0x08)[0]
    scaleY = struct.unpack_from('<f', raw, 0x24)[0]
    imgWidth = struct.unpack_from('<I', raw, 0x3c)[0]
    acqTime = struct.unpack_from('<Q', raw, 0x58)[0]
    imgQuality = 0.0
    if len(raw) >= 0xa0:
        imgQuality = struct.unpack_from('<f', raw, 0x9c)[0]
    return {
        'imgSizeX': imgSizeX, 'imgSizeY': imgSizeY,
        'scaleY': scaleY, 'imgSizeWidth': imgWidth,
        'acquisitionTime': windows_filetime_to_str(acqTime),
        'imageQuality': imgQuality
    }


INVALID_FLOAT = 3.4028234663852886e+38  # FLT_MAX (~NaN marker)
MIN_BM_TRAILING_INVALID = 5  # BM boundary must have at least 5 trailing INVALID values


def count_trailing_invalid(boundary):
    """Count consecutive INVALID values at the end of boundary array."""
    count = 0
    for i in range(len(boundary) - 1, -1, -1):
        if boundary[i] >= INVALID_FLOAT:
            count += 1
        else:
            break
    return count


def export_layers(pat_dir, output_dir):
    """Export layer boundary/thickness CSVs for a single .pat directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Scan all E2E files
    all_seg = []      # (entry, filepath, seg_data)
    all_meta = {}     # (seriesID, imageID) -> metadata
    files = sorted(os.listdir(pat_dir))

    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ('.pdb', '.edb', '.sdb'):
            continue
        filepath = os.path.join(pat_dir, fname)
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            if magic != b'CMDb':
                continue
            f.seek(0x24)
            if f.read(7) != b'MDbMDir':
                continue
            entries = read_mdb_dir(f, 0x4c)
            for e in entries:
                # SegmentationData (0x2723)
                if e['type'] == 0x2723:
                    seg = parse_segmentation(f, e)
                    if seg:
                        all_seg.append({
                            'seriesID': e['seriesID'],
                            'imageID': e['imageID'],
                            'seg': seg,
                            'sourceFile': fname,
                        })
                # BScanMetaData (0x2714) for scaleY
                if e['type'] == 0x2714:
                    meta = parse_bscan_metadata(f, e)
                    if meta:
                        all_meta[(e['seriesID'], e['imageID'])] = meta

    print(f"Found {len(all_seg)} segmentation entries, {len(all_meta)} BScan metadata entries")

    # Group segmentation by (seriesID, imageID)
    bscan_layers = defaultdict(dict)
    for s in all_seg:
        key = (s['seriesID'], s['imageID'])
        seg = s['seg']
        bscan_layers[key][seg['segType']] = {
            'boundary': seg['boundary'],
            'numElements': seg['numElements'],
            'segIndex': seg['segIndex'],
            'sourceFile': s['sourceFile'],
        }

    # ============================================================
    # CSV 1: Layer boundaries (ILM, BM per A-scan)
    # ============================================================
    layer_rows = []
    for (ser, img), layers in sorted(bscan_layers.items()):
        meta = all_meta.get((ser, img), {})
        scaleY_mm = meta.get('scaleY', 0)
        scaleY_um = scaleY_mm * 1000  # mm -> um

        for seg_type, layer_data in sorted(layers.items()):
            layer_name = LAYER_NAMES.get(seg_type, f'Type{seg_type}')
            boundary = layer_data['boundary']
            valid_vals = [v for v in boundary if v < INVALID_FLOAT]
            valid_count = len(valid_vals)
            total = len(boundary)

            row = {
                'seriesID': ser,
                'imageID': img,
                'layerType': seg_type,
                'layerName': layer_name,
                'numAscans': total,
                'validAscans': valid_count,
                'sourceFile': layer_data['sourceFile'],
            }

            if valid_vals:
                mean_px = sum(valid_vals) / len(valid_vals)
                row['meanY_pixel'] = f'{mean_px:.4f}'
                row['minY_pixel'] = f'{min(valid_vals):.4f}'
                row['maxY_pixel'] = f'{max(valid_vals):.4f}'
                if scaleY_um > 0:
                    row['meanY_um'] = f'{mean_px * scaleY_um:.2f}'
            else:
                row['meanY_pixel'] = ''
                row['minY_pixel'] = ''
                row['maxY_pixel'] = ''
                row['meanY_um'] = ''

            # Store full boundary as semicolon-separated values
            row['boundaryData'] = ';'.join(
                f'{v:.4f}' if v < INVALID_FLOAT else 'NaN' for v in boundary
            )
            layer_rows.append(row)

    # ============================================================
    # CSV 2: Thickness per BScan (ILM-to-BM)
    # ============================================================
    thickness_rows = []
    for (ser, img), layers in sorted(bscan_layers.items()):
        meta = all_meta.get((ser, img), {})
        scaleY_mm = meta.get('scaleY', 0)
        scaleY_um = scaleY_mm * 1000
        imgQuality = meta.get('imageQuality', 0)
        acqTime = meta.get('acquisitionTime', '')

        # Find ILM (type=5) and BM (type=2)
        ilm_layer = layers.get(5)
        bm_layer = layers.get(2)

        if not ilm_layer or not bm_layer:
            continue

        ilm = ilm_layer['boundary']
        bm = bm_layer['boundary']
        n = min(len(ilm), len(bm))

        # Apply BM trailing-invalid trimming rule
        bm_trailing = count_trailing_invalid(bm)
        trim_end = max(0, MIN_BM_TRAILING_INVALID - bm_trailing)

        # Get both-valid indices
        bvi = [i for i in range(n) if ilm[i] < INVALID_FLOAT and bm[i] < INVALID_FLOAT]
        if trim_end > 0 and len(bvi) > trim_end:
            bvi = bvi[:len(bvi) - trim_end]

        thickness_px_list = []
        valid_ilm_vals = []
        valid_bm_vals = []
        valid_set = set(bvi)
        valid_count = len(bvi)
        for i in range(n):
            if i in valid_set:
                thickness_px_list.append(bm[i] - ilm[i])
                valid_ilm_vals.append(ilm[i])
                valid_bm_vals.append(bm[i])
            else:
                thickness_px_list.append(None)

        valid_thicknesses = [t for t in thickness_px_list if t is not None]
        if not valid_thicknesses:
            continue

        mean_th_px = sum(valid_thicknesses) / len(valid_thicknesses)
        min_th_px = min(valid_thicknesses)
        max_th_px = max(valid_thicknesses)
        # Mean ILM/BM only where BOTH are valid (matches reference behavior)
        mean_ilm_px = sum(valid_ilm_vals) / len(valid_ilm_vals)
        mean_bm_px = sum(valid_bm_vals) / len(valid_bm_vals)

        row = {
            'seriesID': ser,
            'imageID': img,
            'numAscans': n,
            'validAscans': valid_count,
            'Mean_ILM_Y': f'{mean_ilm_px:.2f}',
            'Mean_BM_Y': f'{mean_bm_px:.2f}',
            'meanThickness_pixel': f'{mean_th_px:.2f}',
            'minThickness_pixel': f'{min_th_px:.2f}',
            'maxThickness_pixel': f'{max_th_px:.2f}',
            'acquisitionTime': acqTime,
            'imageQuality': f'{imgQuality:.2f}',
            'sourceFile': ilm_layer['sourceFile'],
        }

        if scaleY_um > 0:
            mean_th_um = mean_th_px * scaleY_um
            min_th_um = min_th_px * scaleY_um
            max_th_um = max_th_px * scaleY_um
            row['meanThickness_um'] = f'{mean_th_um:.2f}'
            row['minThickness_um'] = f'{min_th_um:.2f}'
            row['maxThickness_um'] = f'{max_th_um:.2f}'
            row['scaleY_mm_per_pixel'] = f'{scaleY_mm:.10f}'

        # Per-Ascan thickness data
        row['thicknessData_pixel'] = ';'.join(
            f'{t:.4f}' if t is not None else 'NaN' for t in thickness_px_list
        )
        if scaleY_um > 0:
            row['thicknessData_um'] = ';'.join(
                f'{t * scaleY_um:.2f}' if t is not None else 'NaN' for t in thickness_px_list
            )

        thickness_rows.append(row)

    # ============================================================
    # CSV 3: Per-Ascan detailed data (ILM, BM, thickness for EVERY A-scan)
    # ============================================================
    ascan_rows = []
    for (ser, img), layers in sorted(bscan_layers.items()):
        meta = all_meta.get((ser, img), {})
        scaleY_mm = meta.get('scaleY', 0)
        scaleY_um = scaleY_mm * 1000

        ilm_layer = layers.get(5)
        bm_layer = layers.get(2)
        if not ilm_layer or not bm_layer:
            continue

        ilm = ilm_layer['boundary']
        bm = bm_layer['boundary']
        n = min(len(ilm), len(bm))

        # Apply same BM trailing-invalid trimming
        bm_trailing = count_trailing_invalid(bm)
        trim_end = max(0, MIN_BM_TRAILING_INVALID - bm_trailing)
        bvi = [i for i in range(n) if ilm[i] < INVALID_FLOAT and bm[i] < INVALID_FLOAT]
        if trim_end > 0 and len(bvi) > trim_end:
            bvi = bvi[:len(bvi) - trim_end]
        valid_set = set(bvi)

        for i in range(n):
            ilm_valid = ilm[i] < INVALID_FLOAT
            bm_valid = bm[i] < INVALID_FLOAT
            is_valid = i in valid_set

            row = {
                'seriesID': ser,
                'imageID': img,
                'ascanIndex': i,
                'ILM_pixel': f'{ilm[i]:.4f}' if ilm_valid else 'NaN',
                'BM_pixel': f'{bm[i]:.4f}' if bm_valid else 'NaN',
            }

            if is_valid:
                th_px = bm[i] - ilm[i]
                row['thickness_pixel'] = f'{th_px:.4f}'
                row['isValid'] = 1
                if scaleY_um > 0:
                    row['ILM_um'] = f'{ilm[i] * scaleY_um:.2f}'
                    row['BM_um'] = f'{bm[i] * scaleY_um:.2f}'
                    row['thickness_um'] = f'{th_px * scaleY_um:.2f}'
            else:
                row['thickness_pixel'] = 'NaN'
                row['isValid'] = 0
                row['ILM_um'] = 'NaN'
                row['BM_um'] = 'NaN'
                row['thickness_um'] = 'NaN'

            ascan_rows.append(row)

    # ============================================================
    # Write CSVs
    # ============================================================
    def write_csv(filename, rows, fieldnames=None):
        if not rows:
            print(f"  {filename}: (no data)")
            return
        if fieldnames is None:
            all_keys = []
            seen = set()
            for row in rows:
                for k in row.keys():
                    if k not in seen:
                        all_keys.append(k)
                        seen.add(k)
            fieldnames = all_keys
        path = os.path.join(output_dir, filename)
        with open(path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"  {filename}: {len(rows)} rows written")

    print(f"\nExporting layer analysis CSVs to: {output_dir}\n")
    write_csv('12_layer_boundaries.csv', layer_rows)
    write_csv('13_thickness_summary.csv', thickness_rows)
    write_csv('14_ascan_detail.csv', ascan_rows)

    print("\nDone!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export retinal layer boundaries & thickness')
    parser.add_argument('pat_dirs', nargs='*',
                        help='.pat directories to process (default: ./00000004.pat)')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output base directory')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    if not args.pat_dirs:
        pat_dir = os.path.join(base, "00000004.pat")
        output_dir = args.output_dir or os.path.join(base, "csv_output")
        print(f"Source: {pat_dir}")
        export_layers(pat_dir, output_dir)
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
                out = os.path.join(base, "batch_output", pat_name)
            print(f"\n{'='*60}")
            print(f"Processing layers: {pd}")
            print(f"{'='*60}")
            export_layers(pd, out)


if __name__ == '__main__':
    main()
