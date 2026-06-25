#!/usr/bin/env python3
"""
Export image metadata CSV: ImageHeader details for every image
(BScan, SLO, Thumbnail, etc.) from E2E files.

Produces: 18_image_details.csv
"""
import struct
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_e2e_csv import scan_all_files, data_content_offset, DATA_ENTRY_HEADER_SIZE

# ImageHeader (from image.cpp):
#   undef[4]  u1(1)  u2(1)  type(1)  u4(1)  u5(4)  breite(4)  hoehe(4)
# Total = 20 bytes
IMAGE_HEADER_SIZE = 20

# Image type IDs from E2E format
IMAGE_TYPE_IDS = {
    0x40000000: 'Image_BScan_SLO',
    0x4000275d: 'Angio_Image',
    0x02:       'Thumbnail_JFIF',
    0x40002710: 'Unknown_0x40002710',
    0x40002711: 'Unknown_0x40002711',
    0x40002712: 'Unknown_0x40002712',
}

# Pixel format names
PIXEL_FMT = {
    (1, None):  ('uint8',  1),  # CV_8UC1
    (32, None): ('uint16', 2),  # CV_16UC1
}


def parse_image_header(f, entry):
    """Read ImageHeader from an image entry. Returns dict or None."""
    content_off = data_content_offset(entry)
    f.seek(content_off)
    raw = f.read(IMAGE_HEADER_SIZE)
    if len(raw) < IMAGE_HEADER_SIZE:
        return None

    undef = raw[:4]
    u1, u2, img_type, u4 = struct.unpack_from('BBBB', raw, 4)
    u5, breite, hoehe = struct.unpack_from('<III', raw, 8)

    # Determine pixel format
    if img_type == 1:
        pixel_fmt = 'uint8'
        bpp = 1
    elif img_type == 32:
        pixel_fmt = 'uint16'
        bpp = 2
    else:
        # Default: SLO=uint8, BScan=uint16
        if entry['subID'] == 0:
            pixel_fmt = 'uint8'
            bpp = 1
        else:
            pixel_fmt = 'uint16'
            bpp = 2

    data_len = entry['dataLength']
    pixel_bytes = data_len - IMAGE_HEADER_SIZE
    expected_bytes = breite * hoehe * bpp
    pixel_offset = entry['dataAddress'] + DATA_ENTRY_HEADER_SIZE + IMAGE_HEADER_SIZE

    return {
        'breite': breite,
        'hoehe': hoehe,
        'img_type': img_type,
        'pixel_fmt': pixel_fmt,
        'bpp': bpp,
        'pixel_bytes': pixel_bytes,
        'expected_bytes': expected_bytes,
        'size_match': pixel_bytes == expected_bytes,
        'pixel_offset': pixel_offset,
        'u1': u1, 'u2': u2, 'u4': u4, 'u5': u5,
    }


def export_image_details(pat_dir, output_dir):
    """Export 18_image_details.csv with ImageHeader data for all images."""
    os.makedirs(output_dir, exist_ok=True)
    all_entries = scan_all_files(pat_dir)

    rows = []
    for entry, filepath in all_entries:
        etype = entry['type']
        # Only process image-type entries
        if etype not in IMAGE_TYPE_IDS and etype != 0x40000000:
            continue

        # Skip JFIF thumbnails (different format - not raw pixels)
        if etype == 0x02:
            continue

        subid = entry['subID']
        if subid == 0:
            img_role = 'SLO'
        elif subid == 1:
            img_role = 'BScan'
        else:
            img_role = f'sub{subid}'

        with open(filepath, 'rb') as f:
            hdr = parse_image_header(f, entry)

        if hdr is None:
            continue

        rows.append({
            'patientID': entry['patientID'],
            'studyID': entry['studyID'],
            'seriesID': entry['seriesID'],
            'imageID': entry['imageID'],
            'subID': subid,
            'imageRole': img_role,
            'typeHex': f"0x{etype:08x}",
            'typeName': IMAGE_TYPE_IDS.get(etype, f'Unknown_{etype}'),
            'breite_height': hdr['breite'],
            'hoehe_width': hdr['hoehe'],
            'img_type': hdr['img_type'],
            'pixel_fmt': hdr['pixel_fmt'],
            'bpp': hdr['bpp'],
            'pixel_bytes': hdr['pixel_bytes'],
            'expected_bytes': hdr['expected_bytes'],
            'size_match': hdr['size_match'],
            'pixel_offset_in_file': hdr['pixel_offset'],
            'sourceFile': entry['_sourceFile'],
        })

    # Write CSV
    csv_path = os.path.join(output_dir, '18_image_details.csv')
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(f"  18_image_details.csv: {len(rows)} rows written")

    # Print summary
    from collections import Counter
    role_count = Counter((r['imageRole'], r['breite_height'], r['hoehe_width'], r['pixel_fmt']) for r in rows)
    print(f"\n  Image Summary:")
    for (role, h, w, fmt), cnt in sorted(role_count.items()):
        print(f"    {role:8s} {h}x{w} {fmt:6s}: {cnt} images")

    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export image details CSV')
    parser.add_argument('pat_dirs', nargs='*',
                        help='.pat directories (default: ./00000004.pat)')
    parser.add_argument('-o', '--output-dir', default=None)
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    if not args.pat_dirs:
        pat_dir = os.path.join(base, '00000004.pat')
        output_dir = args.output_dir or os.path.join(base, 'csv_output')
        print(f"Source: {pat_dir}")
        export_image_details(pat_dir, output_dir)
    else:
        for pd in args.pat_dirs:
            pd = os.path.abspath(pd)
            if not os.path.isdir(pd):
                print(f"Warning: {pd} not found, skip")
                continue
            pat_name = os.path.basename(pd).replace('.pat', '')
            out = args.output_dir or os.path.join(base, 'batch_output', pat_name)
            export_image_details(pd, out)


if __name__ == '__main__':
    main()
