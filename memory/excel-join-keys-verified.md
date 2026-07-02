---
name: excel-join-keys-verified
description: "實際解析兩份治療 Excel 的 join 欄位（2026-07-01）：EYLEA 病歷號(A欄)、Pool Chart no.(D欄) 都是 7–8 位醫院病歷號；推翻舊記憶『pool Chart no. 是 5 碼別序號勿用』。"
metadata:
  type: project
---

2026-07-01 直接解 xlsx XML 驗證兩份治療 Excel 的 join 欄位（為 NAS cohort 篩選）：

- **EYLEA `重新整理過` sheet**：join 欄 = **A 欄 `病歷號`**（存成 shared string）。值為 7–8 位（少數 6 位）醫院病歷號，乾淨數字無連字號（如 3648333 / 39196560 / 8796107）。長度分布：6位×6、7位×26、8位×29。EYLEA 檔**沒有** "Chart no." 欄。
- **Pool `data collection` sheet**：join 欄 = **D 欄 `Chart no.`**（數值）。長度分布：**7位×529、8位×258**、6位×33、5位×7、3位×1 → **絕大多數是 7–8 位醫院病歷號**，與 EYLEA `病歷號`、`.pdb` surname 解出的病歷號同系統。
- **Pool `Name`(E 欄) 放真實姓名**（古彩琴/蔡武雄…），**此版本未去識別化**。

**⚠️ 推翻舊記憶**：[[dataset-ilm-bm]] 與 `CLAUDE.md §7` 寫「case-pooling 的 Chart no.（10160 那種 5 碼）是別的序號，勿用；真病歷號被去識別化進 Name 欄」。但 `20250811更新` 這版**不是那樣**：D 欄 Chart no. 多數就是 7–8 位真病歷號，Name 欄是真姓名。舊說法可能只看了前幾列（前幾列剛好是 5–6 碼舊號）。

**How to apply**：cohort 篩選兩欄都當 key（抓 7–8 位整數），靠「與 `.pdb` 掃出的病歷號取交集」自我驗證雜訊自然落空 → `pipeline/cohort_list_standalone.py` 現行 `^\d{7,8}$` 即可命中兩欄。要納入 6 位少數列就放寬為 `^\d{6,8}$`。相關 pipeline 與產出見 `pipeline/README.md`。
