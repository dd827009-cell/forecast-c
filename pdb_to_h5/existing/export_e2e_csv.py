#!/usr/bin/env python3
"""
E2E File Reader & CSV Exporter
Based on exact LibE2E C++ source code analysis.
Reads HEYEX (Heidelberg Engineering) E2E files and exports all data to CSV.
"""

import struct
import os
import sys
import csv
import datetime
from collections import defaultdict

# ============================================================
# Constants
# ============================================================

# Dir entry Raw struct: 44 bytes (packed)
DIR_ENTRY_FORMAT = '<IIIIiiiihhII'
DIR_ENTRY_SIZE = 44  # struct.calcsize(DIR_ENTRY_FORMAT)

# DataRawHeader: 12 bytes  (8-byte "MDbData\0" + uint32 zero1 + uint32 checksumDatafield)
# But actually it's 8 + 4 + 4 = 16? Let's check: "MDbData" + \0 = 8, uint32 = 4, uint32 = 4 => total 16
# Wait, the C++ struct is: uint8_t[0x08], uint32, uint32 => 8+4+4 = 16
DATA_RAW_HEADER_SIZE = 16
DATA_ENTRY_HEADER_SIZE = DATA_RAW_HEADER_SIZE + DIR_ENTRY_SIZE  # 16 + 44 = 60

# Type ID -> name mapping
TYPE_NAMES = {
    0x40000000: "Image_BScan_SLO",
    0x4000275d: "Angio_Image",
    0x02:       "Thumbnail_JFIF",
    0x07:       "EyeData",
    0x09:       "PatientData",
    0x0a:       "CaptureModule",
    0x0d:       "SpectalisOCT",
    0x11:       "Diagnose",
    0x34:       "PatientUID",
    0x35:       "StudyUID",
    0x36:       "SeriesUID",
    0x3a:       "StudyData",
    0xe8:       "UUID",
    0xe9:       "UnknownID",
    0x2328:     "StudyName",
    0x2329:     "DeviceName",
    0x232d:     "ExaminedStructure",
    0x232e:     "ScanPattern",
    0x232f:     "InfraRed_IR",
    0x2330:     "OCT",
    0x2334:     "Ancestry",
    0x2335:     "PatImage",
    0x2714:     "BScanMetaData",
    0x271c:     "ImageRegistration",
    0x271d:     "BScansMetaData",
    0x2723:     "SegmentationData",
    0x2726:     "ImageRegistration_Star",
    0x2729:     "SloDataElement",
}

# BScanMetaData packed struct (from bscanmetadataelement.cpp)
# uint32 unknown1, uint32 imgSizeX, uint32 imgSizeY, float posX1, float posY1,
# float posX2, float posY2, uint32 zero1, float unknown2, float scaleY,
# float unknown3One, uint32 zero2, float unknown4[2], uint32 zero3,
# uint32 imgSizeWidth, uint32 numImages, uint32 aktImage, uint32 scantype,
# float centerPosX, float centerPosY, uint32 unknown5_4, uint64 acquisitionTime,
# uint32 unknown6[6], uint32 numAve, uint32 unknown7[8], float imageQuality
BSCAN_META_FORMAT = '<IIIffffIfffffII IIIffIQ 6I I 8I f'
# Simplify: read all as bytes and unpack key fields manually
BSCAN_META_SIZE = 0xA0  # ~160 bytes

# SegHeader: uint32 u0, uint32 index, uint32 type, uint32 size, uint32 zeros[5]
SEG_HEADER_FORMAT = '<IIII5I'
SEG_HEADER_SIZE = struct.calcsize(SEG_HEADER_FORMAT)  # 36 bytes

# ============================================================
# Utility functions
# ============================================================

def windows_date_to_str(win_date):
    """Convert Windows OLE Automation DATE (double, days since 1899-12-30)."""
    try:
        if win_date == 0 or win_date < 1:
            return ""
        base = datetime.datetime(1899, 12, 30)
        delta = datetime.timedelta(days=win_date)
        return (base + delta).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, ValueError, OSError):
        return ""


def windows_filetime_to_str(filetime):
    """Convert Windows FILETIME (100ns since 1601-01-01)."""
    try:
        if filetime == 0:
            return ""
        EPOCH_AS_FILETIME = 116444736000000000
        us = (filetime - EPOCH_AS_FILETIME) // 10
        dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=us)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, ValueError, OSError):
        return ""


# ============================================================
# E2E Directory Parsing
# ============================================================

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


def sign_extend_16_to_32(v):
    """Sign-extend int16 to int32, then treat as uint32 (matches C++ behavior)."""
    v16 = v & 0xFFFF
    if v16 & 0x8000:
        return 0xFFFF0000 | v16
    return v16


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
    pos = act_dir_addr + 0x34  # 0x40-12
    while True:
        e = read_dir_entry(f, pos)
        if e is None or e['indexAddress'] != pos:
            break
        if calc_checksum_dir(e) == e['checksum']:
            entries.append(e)
        pos += DIR_ENTRY_SIZE
    return entries


def get_data_class(e):
    if e['imageID'] != -1: return "Image"
    if e['seriesID'] != -1: return "Series"
    if e['studyID'] != -1: return "Study"
    if e['patientID'] != -1: return "Patient"
    return "General"


def data_content_offset(entry):
    """Return file offset to data content (after MDbData header + Raw header)."""
    return entry['dataAddress'] + DATA_ENTRY_HEADER_SIZE


# ============================================================
# Data Element Parsers (matching C++ source exactly)
# ============================================================

def parse_patient_data(f, entry):
    """PatientDataElement: expects dataLength == 131"""
    if entry['dataLength'] != 131:
        return None
    f.seek(data_content_offset(entry))
    startpos = f.tell()
    raw = f.read(131)
    if len(raw) < 131:
        return None

    # forename: offset 0, max 30 chars
    forename = raw[0:30].split(b'\x00')[0].decode('latin-1', errors='replace')
    # surname: offset 31, max 50 chars
    surname = raw[31:81].split(b'\x00')[0].decode('latin-1', errors='replace')
    # title: offset 82, max 10 chars
    title = raw[82:92].split(b'\x00')[0].decode('latin-1', errors='replace')
    # windowsBirthDate: offset 93, double (8 bytes)
    birth_date = struct.unpack_from('<d', raw, 93)[0]
    # sexChar: offset 101, 1 byte
    sex_char = raw[101]
    sex = 'Female' if sex_char == 0x46 else ('Male' if sex_char == 0x4d else 'Unknown')
    # id: offset 102, max 20 chars
    pat_id = raw[102:122].split(b'\x00')[0].decode('latin-1', errors='replace')

    return {
        'forename': forename, 'surname': surname, 'title': title,
        'birthDate': windows_date_to_str(birth_date), 'sex': sex, 'patientIdStr': pat_id
    }


def parse_study_data(f, entry):
    """StudyData: expects dataLength == 91"""
    if entry['dataLength'] != 91:
        return None
    f.seek(data_content_offset(entry))
    raw = f.read(91)
    if len(raw) < 91:
        return None
    # offset 6: windowsStudyDate (double)
    study_date = struct.unpack_from('<d', raw, 6)[0]
    # offset 16: operator (max 16 chars)
    operator = raw[16:32].split(b'\x00')[0].decode('latin-1', errors='replace')
    return {'operator': operator, 'studyDate': windows_date_to_str(study_date)}


def parse_eye_data(f, entry):
    """EyeData: expects dataLength == 67 or 68"""
    if entry['dataLength'] not in (67, 68):
        return None
    f.seek(data_content_offset(entry))
    raw = f.read(entry['dataLength'])
    if len(raw) < 67:
        return None
    # Read exactly as C++ does: eyeSide(char), iop, refraction, c_curve,
    # vfieldMean, vfieldVar, cylinder, axis, correctiveLens(uint16), pupilSize
    eye_side_byte = raw[0]
    eye_side = chr(eye_side_byte) if eye_side_byte in (ord('L'), ord('R')) else f'0x{eye_side_byte:02x}'
    iop      = struct.unpack_from('<d', raw, 1)[0]
    refract  = struct.unpack_from('<d', raw, 9)[0]
    c_curve  = struct.unpack_from('<d', raw, 17)[0]
    vf_mean  = struct.unpack_from('<d', raw, 25)[0]
    vf_var   = struct.unpack_from('<d', raw, 33)[0]
    cylinder = struct.unpack_from('<d', raw, 41)[0]
    axis     = struct.unpack_from('<d', raw, 49)[0]
    corr_lens = struct.unpack_from('<H', raw, 57)[0]
    pupil    = struct.unpack_from('<d', raw, 59)[0]

    return {
        'eyeSide': eye_side, 'iop_mmHg': iop, 'refraction_dpt': refract,
        'c_curve_mm': c_curve, 'vfieldMean': vf_mean, 'vfieldVar': vf_var,
        'cylinder_dpt': cylinder, 'axis_deg': axis, 'correctiveLens': corr_lens,
        'pupilSize_mm': pupil
    }


def parse_text_element(f, entry):
    """TextElement: null-terminated string."""
    f.seek(data_content_offset(entry))
    raw = f.read(entry['dataLength'])
    text = raw.split(b'\x00')[0].decode('latin-1', errors='replace')
    return text


def parse_text_element16(f, entry):
    """TextElement16: UTF-16LE string."""
    f.seek(data_content_offset(entry))
    raw = f.read(entry['dataLength'])
    try:
        text = raw.decode('utf-16-le', errors='replace').rstrip('\x00')
    except:
        text = ""
    return text


def parse_string_list(f, entry):
    """StringListElement: header(uint32 stringNumbers, uint32 stringSize) + strings."""
    f.seek(data_content_offset(entry))
    raw = f.read(entry['dataLength'])
    if len(raw) < 8:
        return []
    string_numbers, string_size = struct.unpack_from('<II', raw, 0)
    if string_numbers * string_size + 8 != entry['dataLength']:
        return []  # validation like C++
    result = []
    for i in range(string_numbers):
        offset = 8 + string_size * i
        chunk = raw[offset:offset + string_size]
        # Read UTF-16 chars until null
        chars = []
        for c in range(0, len(chunk) - 1, 2):
            ch = struct.unpack_from('<H', chunk, c)[0]
            if ch == 0:
                break
            chars.append(chr(ch))
        result.append(''.join(chars))
    return result


def parse_bscan_metadata(f, entry):
    """BScanMetaDataElement: MetaDataStruct packed struct."""
    f.seek(data_content_offset(entry))
    raw = f.read(min(entry['dataLength'], 256))
    if len(raw) < 0x60:
        return None

    # Exact layout from bscanmetadataelement.cpp MetaDataStruct:
    # 0x00: uint32 unknown1
    # 0x04: uint32 imgSizeX
    # 0x08: uint32 imgSizeY
    # 0x0c: float posX1
    # 0x10: float posY1
    # 0x14: float posX2
    # 0x18: float posY2
    # 0x1c: uint32 zero1
    # 0x20: float unknown2
    # 0x24: float scaleY
    # 0x28: float unknown3One
    # 0x2c: uint32 zero2
    # 0x30: float unknown4[2]
    # 0x38: uint32 zero3
    # 0x3c: uint32 imgSizeWidth
    # 0x40: uint32 numImages
    # 0x44: uint32 aktImage
    # 0x48: uint32 scantype
    # 0x4c: float centerPosX
    # 0x50: float centerPosY
    # 0x54: uint32 unknown5_4
    # 0x58: uint64 acquisitionTime
    # ... uint32 unknown6[6], uint32 numAve, uint32 unknown7[8], float imageQuality

    imgSizeX = struct.unpack_from('<I', raw, 0x04)[0]
    imgSizeY = struct.unpack_from('<I', raw, 0x08)[0]
    posX1    = struct.unpack_from('<f', raw, 0x0c)[0]
    posY1    = struct.unpack_from('<f', raw, 0x10)[0]
    posX2    = struct.unpack_from('<f', raw, 0x14)[0]
    posY2    = struct.unpack_from('<f', raw, 0x18)[0]
    scaleY   = struct.unpack_from('<f', raw, 0x24)[0]
    imgWidth = struct.unpack_from('<I', raw, 0x3c)[0]
    numImg   = struct.unpack_from('<I', raw, 0x40)[0]
    aktImg   = struct.unpack_from('<I', raw, 0x44)[0]
    scantype = struct.unpack_from('<I', raw, 0x48)[0]
    centerX  = struct.unpack_from('<f', raw, 0x4c)[0]
    centerY  = struct.unpack_from('<f', raw, 0x50)[0]
    acqTime  = struct.unpack_from('<Q', raw, 0x58)[0]

    scantype_name = {0: 'Unknown', 1: 'Line/Star', 2: 'Circle'}.get(scantype, f'Unknown({scantype})')

    # numAve at offset 0x58+8+6*4 = 0x78
    numAve = 0
    imgQuality = 0.0
    if len(raw) >= 0x7c:
        numAve = struct.unpack_from('<I', raw, 0x78)[0]
    if len(raw) >= 0xa0:
        imgQuality = struct.unpack_from('<f', raw, 0x9c)[0]

    return {
        'imgSizeX': imgSizeX, 'imgSizeY': imgSizeY,
        'posX1': posX1, 'posY1': posY1, 'posX2': posX2, 'posY2': posY2,
        'scaleY': scaleY, 'imgSizeWidth': imgWidth,
        'numImages': numImg, 'aktImage': aktImg,
        'scanType': scantype_name, 'centerX': centerX, 'centerY': centerY,
        'acquisitionTime': windows_filetime_to_str(acqTime),
        'numAve': numAve, 'imageQuality': imgQuality
    }


def parse_bscans_metadata(f, entry):
    """BScansMetaDataElement: Header(uint32[3] undef, uint32 numImages) + RawData[] entries."""
    f.seek(data_content_offset(entry))
    raw = f.read(entry['dataLength'])
    if len(raw) < 16:
        return []
    numImages = struct.unpack_from('<I', raw, 12)[0]
    # RawData size: uint32[12]+float[4]+uint32[5] = 48+16+20 = 84 bytes
    RAW_DATA_SIZE = 84
    result = []
    for i in range(numImages):
        offset = 16 + RAW_DATA_SIZE * i
        if offset + RAW_DATA_SIZE > len(raw):
            break
        # x1,y1,x2,y2 at offset 48 (12*4) from RawData start
        x1 = struct.unpack_from('<f', raw, offset + 48)[0]
        y1 = struct.unpack_from('<f', raw, offset + 52)[0]
        x2 = struct.unpack_from('<f', raw, offset + 56)[0]
        y2 = struct.unpack_from('<f', raw, offset + 60)[0]
        result.append({'scanIndex': i, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    return result


def parse_segmentation_data(f, entry):
    """SegmentationData: SegHeader + float array."""
    f.seek(data_content_offset(entry))
    raw = f.read(entry['dataLength'])
    if len(raw) < SEG_HEADER_SIZE:
        return None
    vals = struct.unpack_from(SEG_HEADER_FORMAT, raw, 0)
    u0, index, seg_type, size = vals[0], vals[1], vals[2], vals[3]
    max_elements = (entry['dataLength'] - SEG_HEADER_SIZE) // 4
    num_elements = min(size, max_elements)
    seg_data = []
    for i in range(num_elements):
        offset = SEG_HEADER_SIZE + i * 4
        if offset + 4 > len(raw):
            break
        seg_data.append(struct.unpack_from('<f', raw, offset)[0])
    return {'segIndex': index, 'segType': seg_type, 'numElements': num_elements, 'data': seg_data}


def parse_image_registration(f, entry):
    """ImageRegistration: 25 floats, dataLength == 100."""
    if entry['dataLength'] != 100:
        return None
    f.seek(data_content_offset(entry))
    raw = f.read(100)
    if len(raw) < 100:
        return None
    values = struct.unpack_from('<25f', raw, 0)
    return list(values)


def parse_slo_data(f, entry):
    """SloDataElement: skip 24 bytes, then uint64 winDate, then 6 floats."""
    f.seek(data_content_offset(entry))
    raw = f.read(entry['dataLength'])
    if len(raw) < 56:  # 24 + 8 + 24
        return None
    win_date = struct.unpack_from('<Q', raw, 24)[0]
    transform = list(struct.unpack_from('<6f', raw, 32))
    return {'date': windows_filetime_to_str(win_date), 'transform': transform}


def parse_image_info(f, entry):
    """Image header: just extract basic dimensions info."""
    f.seek(data_content_offset(entry))
    raw = f.read(min(entry['dataLength'], 64))
    if len(raw) < 12:
        return None
    # The C++ Image constructor reads an image header then OpenCV image data
    # We can at least report size
    return {'dataSizeBytes': entry['dataLength']}


# ============================================================
# Main scan + export logic
# ============================================================

def scan_all_files(pat_dir):
    """Scan all E2E files and return parsed data grouped by hierarchy."""
    files = sorted(os.listdir(pat_dir))
    all_entries = []

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
                e['_sourceFile'] = fname
                e['_sourceExt'] = ext
                all_entries.append((e, filepath))

    return all_entries


def export_csvs(pat_dir, output_dir):
    """Export all data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    all_entries = scan_all_files(pat_dir)

    print(f"Total entries across all files: {len(all_entries)}")

    # Collect data for each CSV
    patient_rows = []
    study_rows = []
    eye_data_rows = []
    series_rows = []
    bscan_meta_rows = []
    bscans_meta_rows = []
    segmentation_rows = []
    image_reg_rows = []
    image_rows = []
    slo_data_rows = []
    all_entries_rows = []

    file_handles = {}

    for entry, filepath in all_entries:
        if filepath not in file_handles:
            file_handles[filepath] = open(filepath, 'rb')
        f = file_handles[filepath]

        type_id = entry['type']
        type_name = TYPE_NAMES.get(type_id, f'Unknown_0x{type_id:08x}')
        data_class = get_data_class(entry)

        # Universal entry row
        all_entries_rows.append({
            'sourceFile': entry['_sourceFile'],
            'fileType': entry['_sourceExt'],
            'typeId': f'0x{type_id:08x}',
            'typeName': type_name,
            'dataClass': data_class,
            'patientID': entry['patientID'],
            'studyID': entry['studyID'],
            'seriesID': entry['seriesID'],
            'imageID': entry['imageID'],
            'subID': entry['subID'],
            'dataLength': entry['dataLength'],
            'dataAddress': f'0x{entry["dataAddress"]:08x}',
        })

        try:
            # Patient Data (0x09)
            if type_id == 0x09:
                pd = parse_patient_data(f, entry)
                if pd:
                    pd['patientID'] = entry['patientID']
                    pd['sourceFile'] = entry['_sourceFile']
                    patient_rows.append(pd)

            # PatientUID (0x34)
            elif type_id == 0x34:
                uid = parse_text_element(f, entry)
                patient_rows.append({
                    'patientID': entry['patientID'],
                    'sourceFile': entry['_sourceFile'],
                    'patientUID': uid
                })

            # Diagnose (0x11)
            elif type_id == 0x11:
                diag = parse_text_element16(f, entry)
                patient_rows.append({
                    'patientID': entry['patientID'],
                    'sourceFile': entry['_sourceFile'],
                    'diagnose': diag
                })

            # Ancestry (0x2334)
            elif type_id == 0x2334:
                strings = parse_string_list(f, entry)
                patient_rows.append({
                    'patientID': entry['patientID'],
                    'sourceFile': entry['_sourceFile'],
                    'ancestry': '; '.join(strings)
                })

            # StudyData (0x3a)
            elif type_id == 0x3a:
                sd = parse_study_data(f, entry)
                if sd:
                    sd['patientID'] = entry['patientID']
                    sd['studyID'] = entry['studyID']
                    sd['sourceFile'] = entry['_sourceFile']
                    study_rows.append(sd)

            # StudyUID (0x35)
            elif type_id == 0x35:
                uid = parse_text_element(f, entry)
                study_rows.append({
                    'patientID': entry['patientID'],
                    'studyID': entry['studyID'],
                    'sourceFile': entry['_sourceFile'],
                    'studyUID': uid
                })

            # StudyName (0x2328)
            elif type_id == 0x2328:
                strings = parse_string_list(f, entry)
                study_rows.append({
                    'patientID': entry['patientID'],
                    'studyID': entry['studyID'],
                    'sourceFile': entry['_sourceFile'],
                    'studyName': '; '.join(strings)
                })

            # EyeData (0x07)
            elif type_id == 0x07:
                ed = parse_eye_data(f, entry)
                if ed:
                    ed['patientID'] = entry['patientID']
                    ed['studyID'] = entry['studyID']
                    ed['sourceFile'] = entry['_sourceFile']
                    eye_data_rows.append(ed)

            # SeriesUID (0x36)
            elif type_id == 0x36:
                uid = parse_text_element(f, entry)
                series_rows.append({
                    'patientID': entry['patientID'],
                    'studyID': entry['studyID'],
                    'seriesID': entry['seriesID'],
                    'subID': entry['subID'],
                    'sourceFile': entry['_sourceFile'],
                    'seriesUID': uid
                })

            # ExaminedStructure (0x232d)
            elif type_id == 0x232d:
                strings = parse_string_list(f, entry)
                series_rows.append({
                    'patientID': entry['patientID'],
                    'studyID': entry['studyID'],
                    'seriesID': entry['seriesID'],
                    'sourceFile': entry['_sourceFile'],
                    'examinedStructure': '; '.join(strings)
                })

            # ScanPattern (0x232e)
            elif type_id == 0x232e:
                strings = parse_string_list(f, entry)
                series_rows.append({
                    'patientID': entry['patientID'],
                    'studyID': entry['studyID'],
                    'seriesID': entry['seriesID'],
                    'sourceFile': entry['_sourceFile'],
                    'scanPattern': '; '.join(strings)
                })

            # DeviceName (0x2329)
            elif type_id == 0x2329:
                strings = parse_string_list(f, entry)
                series_rows.append({
                    'patientID': entry['patientID'],
                    'studyID': entry['studyID'],
                    'seriesID': entry['seriesID'],
                    'sourceFile': entry['_sourceFile'],
                    'deviceName': '; '.join(strings)
                })

            # BScanMetaData (0x2714)
            elif type_id == 0x2714:
                bm = parse_bscan_metadata(f, entry)
                if bm:
                    bm['patientID'] = entry['patientID']
                    bm['studyID'] = entry['studyID']
                    bm['seriesID'] = entry['seriesID']
                    bm['imageID'] = entry['imageID']
                    bm['sourceFile'] = entry['_sourceFile']
                    bscan_meta_rows.append(bm)

            # BScansMetaData (0x271d)
            elif type_id == 0x271d:
                items = parse_bscans_metadata(f, entry)
                for item in items:
                    item['patientID'] = entry['patientID']
                    item['studyID'] = entry['studyID']
                    item['seriesID'] = entry['seriesID']
                    item['sourceFile'] = entry['_sourceFile']
                    bscans_meta_rows.append(item)

            # SegmentationData (0x2723)
            elif type_id == 0x2723:
                seg = parse_segmentation_data(f, entry)
                if seg:
                    seg_row = {
                        'patientID': entry['patientID'],
                        'studyID': entry['studyID'],
                        'seriesID': entry['seriesID'],
                        'imageID': entry['imageID'],
                        'sourceFile': entry['_sourceFile'],
                        'segIndex': seg['segIndex'],
                        'segType': seg['segType'],
                        'numElements': seg['numElements'],
                        'segData': ';'.join(f'{v:.4f}' for v in seg['data'])
                    }
                    segmentation_rows.append(seg_row)

            # ImageRegistration (0x271c, 0x2726)
            elif type_id in (0x271c, 0x2726):
                vals = parse_image_registration(f, entry)
                if vals:
                    image_reg_rows.append({
                        'patientID': entry['patientID'],
                        'studyID': entry['studyID'],
                        'seriesID': entry['seriesID'],
                        'imageID': entry['imageID'],
                        'sourceFile': entry['_sourceFile'],
                        'typeId': f'0x{type_id:08x}',
                        'values': ';'.join(f'{v:.6f}' for v in vals)
                    })

            # Image (0x40000000, 0x4000275d)
            elif type_id in (0x40000000, 0x4000275d):
                info = parse_image_info(f, entry)
                if info:
                    image_rows.append({
                        'patientID': entry['patientID'],
                        'studyID': entry['studyID'],
                        'seriesID': entry['seriesID'],
                        'imageID': entry['imageID'],
                        'subID': entry['subID'],
                        'sourceFile': entry['_sourceFile'],
                        'imageType': 'Angio' if type_id == 0x4000275d else ('SLO' if data_class == 'Series' or (entry['subID'] == 0 and data_class == 'Image') else 'BScan'),
                        'dataSizeBytes': info['dataSizeBytes']
                    })

            # SloDataElement (0x2729)
            elif type_id == 0x2729:
                slo = parse_slo_data(f, entry)
                if slo:
                    slo['patientID'] = entry['patientID']
                    slo['studyID'] = entry['studyID']
                    slo['seriesID'] = entry['seriesID']
                    slo['sourceFile'] = entry['_sourceFile']
                    slo['transform'] = ';'.join(f'{v:.6f}' for v in slo['transform'])
                    slo_data_rows.append(slo)

            # Thumbnail (0x02)
            elif type_id == 0x02:
                image_rows.append({
                    'patientID': entry['patientID'],
                    'studyID': entry['studyID'],
                    'seriesID': entry['seriesID'],
                    'imageID': entry['imageID'],
                    'subID': entry['subID'],
                    'sourceFile': entry['_sourceFile'],
                    'imageType': 'Thumbnail_JFIF',
                    'dataSizeBytes': entry['dataLength']
                })

        except Exception as ex:
            pass  # Skip unparseable entries

    # Close file handles
    for fh in file_handles.values():
        fh.close()

    # ============================================================
    # Write CSVs
    # ============================================================

    def write_csv(filename, rows, fieldnames=None):
        if not rows:
            print(f"  {filename}: (no data)")
            return
        if fieldnames is None:
            # Collect all keys
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

    print(f"\nExporting CSVs to: {output_dir}\n")

    write_csv('01_all_entries.csv', all_entries_rows)
    write_csv('02_patient_data.csv', patient_rows)
    write_csv('03_study_data.csv', study_rows)
    write_csv('04_eye_data.csv', eye_data_rows)
    write_csv('05_series_info.csv', series_rows)
    write_csv('06_bscan_metadata.csv', bscan_meta_rows)
    write_csv('07_bscans_positions.csv', bscans_meta_rows)
    write_csv('08_segmentation.csv', segmentation_rows)
    write_csv('09_image_registration.csv', image_reg_rows)
    write_csv('10_images.csv', image_rows)
    write_csv('11_slo_data.csv', slo_data_rows)

    print("\nDone!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='E2E File Reader & CSV Exporter')
    parser.add_argument('pat_dirs', nargs='*',
                        help='.pat directories to process (default: ./00000004.pat)')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output base directory (batch: output/<pat_name>)')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    if not args.pat_dirs:
        # Single-dir mode (backward compatible)
        pat_dir = os.path.join(base, "00000004.pat")
        output_dir = args.output_dir or os.path.join(base, "csv_output")
        if not os.path.isdir(pat_dir):
            print(f"Error: Directory not found: {pat_dir}")
            sys.exit(1)
        print(f"Source: {pat_dir}")
        export_csvs(pat_dir, output_dir)
    else:
        # Batch mode
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
            print(f"Processing: {pd}")
            print(f"Output:     {out}")
            print(f"{'='*60}")
            export_csvs(pd, out)


if __name__ == '__main__':
    main()
