#!/usr/bin/env python3
"""case pooling「粗治療」解析器 + 覆蓋分析。純標準庫(zip 解 xlsx)，免 pandas/torch。

case pooling 有「藥種 + 治療起日 + regimen + 診斷 + baseline」但**沒逐針時間軸**。
→ 提供粗粒度治療條件（哪種 anti-VEGF、距治療起日幾天），補 EYLEA 逐針細治療的不足。

欄位(data collection sheet)：
  D=病歷號(取7-8位) AO=眼(1OD/2OS) AR=施打起日(Excel序列) AZ=Regimen AN=診斷 AV=PED>400
  AW=Eylea AX=Lucentis AY=Avastin (0/1) AP=年齡 AQ=性別

用法: python pipeline/case_pooling.py <case_pooling.xlsx> <patient_stats.csv>
"""
import sys
import re
import csv
import datetime
import zipfile
import collections
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
CHART = re.compile(r"^\d{7,8}$")
COL = {"chart": "D", "eye": "AO", "apply": "AR", "regimen": "AZ", "dx": "AN",
       "ped": "AV", "eylea": "AW", "lucentis": "AX", "avastin": "AY",
       "age": "AP", "sex": "AQ"}


def _excel_date(v):
    try:
        n = int(float(v))
        return (datetime.date(1899, 12, 30) + datetime.timedelta(days=n)).isoformat()
    except (ValueError, TypeError):
        return ""


def parse_case_pooling(xlsx, sheet="data collection"):
    z = zipfile.ZipFile(xlsx)
    ss = ["".join(t.text or "" for t in si.iter(f"{NS}t"))
          for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{NS}si")]
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")).iter()}
    sf = None
    for s in ET.fromstring(z.read("xl/workbook.xml")).iter(f"{NS}sheet"):
        if s.get("name") == sheet:
            sf = "xl/" + rels[s.get(f"{RNS}id")]
    root = ET.fromstring(z.read(sf))

    def cletter(ref):
        return re.match(r"[A-Z]+", ref).group()

    def cval(c):
        t = c.get("t"); v = c.find(f"{NS}v")
        if v is not None and v.text is not None:
            return ss[int(v.text)] if (t == "s" and v.text.isdigit()) else v.text
        isx = c.find(f"{NS}is")
        return "".join(tt.text or "" for tt in isx.iter(f"{NS}t")) if isx is not None else ""

    recs = {}
    for row in root.iter(f"{NS}row"):
        if int(row.get("r")) == 1:
            continue
        cells = {cletter(c.get("r")): cval(c) for c in row.findall(f"{NS}c")}
        d = str(cells.get(COL["chart"], "")).strip().replace(".0", "")
        if not CHART.match(d):
            continue
        eye = "OD" if str(cells.get(COL["eye"], "")).strip().replace(".0", "") == "1" else "OS"
        drugs = [name for name, col in (("Eylea", "eylea"), ("Lucentis", "lucentis"), ("Avastin", "avastin"))
                 if str(cells.get(COL[col], "")).strip() == "1"]
        recs[(d, eye)] = {
            "drugs": drugs,
            "apply_date": _excel_date(cells.get(COL["apply"], "")),
            "regimen": str(cells.get(COL["regimen"], "")).strip(),
            "diagnosis": str(cells.get(COL["dx"], "")).strip().replace(".0", ""),
            "ped_gt400": str(cells.get(COL["ped"], "")).strip().replace(".0", ""),
            "age": str(cells.get(COL["age"], "")).strip(),
            "sex": str(cells.get(COL["sex"], "")).strip().replace(".0", ""),
        }
    return recs


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    xlsx, stats = sys.argv[1], sys.argv[2]
    recs = parse_case_pooling(xlsx)
    print(f"case pooling 解析 {len(recs)} 條 (病歷號,eye) 粗治療")

    h5_eyes = set()
    for r in csv.DictReader(open(stats, encoding="utf-8-sig")):
        c = str(r.get("病歷號", "")).strip()
        if not c:
            continue
        if int(r.get("OD回診日數") or 0) > 0:
            h5_eyes.add((c, "OD"))
        if int(r.get("OS回診日數") or 0) > 0:
            h5_eyes.add((c, "OS"))

    matched = set(recs) & h5_eyes
    print(f"h5 有OCT的眼={len(h5_eyes)}  case pooling 治療眼={len(recs)}  ★對上(有OCT+粗治療)={len(matched)}")

    drug_combo = collections.Counter(); dx = collections.Counter(); has_apply = 0
    for k in matched:
        r = recs[k]
        drug_combo[tuple(r["drugs"]) or ("(無)",)] += 1
        dx[r["diagnosis"]] += 1
        if r["apply_date"]:
            has_apply += 1
    print(f"\n藥種組合分布(對上的眼): {dict(drug_combo.most_common())}")
    print(f"診斷分布(1nAMD/2PCV/3RAP): {dict(dx.most_common())}")
    print(f"有治療起日: {has_apply}/{len(matched)}")


if __name__ == "__main__":
    main()
