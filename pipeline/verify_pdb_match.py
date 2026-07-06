#!/usr/bin/env python3
"""重做 .pdb 病例對照：Excel 各治療分頁病歷號 vs .pdb 掃描(已下載 .pat)。
逐張分頁報「有幾個病歷號、對到 .pdb 幾個、沒對到幾個」，確認 AI整理/nAMD 有無正確抓到。
純標準庫(zip 解 xlsx)。

用法: python pipeline/verify_pdb_match.py <pat_index.csv> <EYLEA.xlsx> <case_pooling.xlsx>
"""
import sys
import csv
import re
import zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
CHART = re.compile(r"^\d{7,8}$")


def _colletter(ref):
    return re.match(r"[A-Z]+", ref).group()


def read_sheet(xlsx, sheetname):
    z = zipfile.ZipFile(xlsx)
    ss = ["".join(t.text or "" for t in si.iter(f"{NS}t"))
          for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{NS}si")]
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")).iter()}
    sf = None
    for s in ET.fromstring(z.read("xl/workbook.xml")).iter(f"{NS}sheet"):
        if s.get("name") == sheetname:
            sf = "xl/" + rels[s.get(f"{RNS}id")]
    root = ET.fromstring(z.read(sf))

    def cval(c):
        t = c.get("t"); v = c.find(f"{NS}v")
        if v is not None and v.text is not None:
            return ss[int(v.text)] if (t == "s" and v.text.isdigit()) else v.text
        isx = c.find(f"{NS}is")
        return "".join(tt.text or "" for tt in isx.iter(f"{NS}t")) if isx is not None else ""

    rows = {}
    for row in root.iter(f"{NS}row"):
        ri = int(row.get("r"))
        for c in row.findall(f"{NS}c"):
            rows.setdefault(ri, {})[_colletter(c.get("r"))] = cval(c)
    return rows


def charts_from_sheet(rows, col_letter=None, header_key=None):
    if header_key and not col_letter:
        for cl, v in rows.get(1, {}).items():
            if header_key in str(v):
                col_letter = cl
                break
    out = set()
    if not col_letter:
        return out, col_letter
    for ri, cells in rows.items():
        if ri == 1:
            continue
        v = str(cells.get(col_letter, "")).strip().replace(".0", "")
        if CHART.match(v):
            out.add(v)
    return out, col_letter


def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)
    idx_csv, eylea, cp = sys.argv[1], sys.argv[2], sys.argv[3]

    pdb = set()
    for r in csv.DictReader(open(idx_csv, encoding="utf-8")):
        c = str(r.get("chart_no", "")).strip()
        if c:
            pdb.add(c)
    print(f".pdb 掃描到的病歷號(已下載 .pat) = {len(pdb)}\n")

    sources = [
        ("EYLEA 重新整理過", eylea, "重新整理過", None, "病歷號"),
        ("EYLEA AI整理", eylea, "AI整理", "A", None),
        ("EYLEA nAMD", eylea, "nAMD", "A", None),
        ("case pooling", cp, "data collection", "D", None),
    ]
    all_tx = set()
    print(f"{'分頁':22s}{'病歷號':>8}{'對到.pdb':>10}{'沒對到':>8}{'欄':>5}")
    for label, path, sheet, col, hkey in sources:
        try:
            rows = read_sheet(path, sheet)
        except Exception as e:
            print(f"  {label}: 讀取失敗 {e}")
            continue
        charts, used_col = charts_from_sheet(rows, col_letter=col, header_key=hkey)
        all_tx |= charts
        m = charts & pdb
        print(f"{label:22s}{len(charts):>8}{len(m):>10}{len(charts - m):>8}{str(used_col):>5}")

    mt = all_tx & pdb
    print(f"\n所有治療分頁合併: 病歷號 {len(all_tx)}｜對到 .pdb {len(mt)}｜沒對到 {len(all_tx - mt)}")
    with open("/mnt/d/tx_charts_not_in_pdb.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(all_tx - pdb)))
    print("『有治療但沒對到已下載.pdb』的病歷號 → D:\\tx_charts_not_in_pdb.txt")


if __name__ == "__main__":
    main()
