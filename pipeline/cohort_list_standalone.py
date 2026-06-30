#!/usr/bin/env python3
"""自含版：從一或多份 .xlsx 抽出所有『7–8 位整數』當病歷號 → cohort.txt。
零安裝（純 Python 標準庫，不需 pandas/openpyxl，不需 repo）。

為什麼用 7–8 位：醫院病歷號就是 7–8 位（EYLEA 的「病歷號」欄、case pooling 去識別化後填進
「Name」欄的，都是這格式）；研究序號只有 4 位、CMT/VA/Excel 日期序號都不是 7–8 位 → 安全。
最後仍由 filter_pats 與 .pdb 掃出的 index.csv 取交集驗證，非病歷號的雜訊自然落空。

用法:
  python cohort_list_standalone.py <輸出cohort.txt> <檔1.xlsx> [檔2.xlsx ...]
例:
  python cohort_list_standalone.py cohort.txt "EYLEA....xlsx" "全體系case pooling....xlsx"
"""
import sys
import re
import zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CHART = re.compile(r"^\d{7,8}$")


def _shared_strings(z):
    try:
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(t.text or "" for t in si.iter(f"{NS}t")) for si in root.findall(f"{NS}si")]


def charts_from_xlsx(path):
    found = set()
    with zipfile.ZipFile(path) as z:
        ss = _shared_strings(z)
        sheets = [n for n in z.namelist()
                  if n.startswith("xl/worksheets/") and n.endswith(".xml")]
        for sn in sheets:
            root = ET.fromstring(z.read(sn))
            for c in root.iter(f"{NS}c"):
                t = c.get("t")
                v = c.find(f"{NS}v")
                if v is not None and v.text is not None:
                    val = ss[int(v.text)] if (t == "s" and v.text.isdigit()) else v.text
                else:  # inline string
                    is_ = c.find(f"{NS}is")
                    val = "".join(tt.text or "" for tt in is_.iter(f"{NS}t")) if is_ is not None else None
                if val is None:
                    continue
                val = str(val).strip()
                if val.endswith(".0") and val[:-2].isdigit():  # 防 float 表示
                    val = val[:-2]
                if CHART.match(val):
                    found.add(val)
    return found


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[1]
    allc = set()
    for p in sys.argv[2:]:
        c = charts_from_xlsx(p)
        print(f"  {p}: {len(c)} 個 7-8 位病歷號")
        allc |= c
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(allc)))
    print(f"共 {len(allc)} 個唯一病歷號 → {out}")


if __name__ == "__main__":
    main()
