# pipeline/ — NAS raw 取得 → cohort 篩選 → h5

> 目標：從 NAS 上海量 `.pat`（主路徑就有 ~99670 個）裡，**只下載治療 cohort 需要的影像**，
> 並為**每個 `.pat` 留一筆 log（病歷號 + `.sdb` 數）**。策略＝**先掃再取**：
> 先列檔 + 只下小的 `.pdb` 解出病歷號 → 對 Excel 取交集 → 才下載命中者的完整大檔（`.sdb`）。

## 0. 密碼安全（先做）
所有腳本都**不碰明碼**，走 `~/.netrc`：
```
machine cad.csie.ntu.edu.tw login d13945010 password 你的密碼
```
然後 `chmod 600 ~/.netrc`。lftp 會自動讀取。

需要 `lftp`（FTP 鏡像）、Python（轉檔 venv，含 `pandas`/`eyepy`；掃描只需標準庫）。

## 1. 一鍵跑全部路徑（建議）
編輯 `run_all_paths.sh` 最上面「設定」5 行（`EXCELS` 兩份檔的實際路徑、`SSD` 工作目錄），然後：
```bash
bash pipeline/run_all_paths.sh
```
它會對 `nas_bases.txt` 每個 base 依序做：列檔 → 推 `.pat` 清單 → 只下 `.pdb` → 掃病歷號 →
per-base `pat_log.csv`；最後對兩份 Excel 篩 cohort → 只下命中者完整 → 轉 h5 → 合併總 `pat_log_all.csv`。
**全程可續跑**（列檔/下載/轉檔都冪等），中斷再跑即可。

### 處理哪些路徑？`nas_bases.txt`
主路徑 `/eye2/eye4(cad5)/ike/patients` ＋ `naspath.txt` 解碼後的 7 條（Big5 百分比編碼已解開，
如 `%A4gOCT`→`土OCT`、`202604%A9%FA%AA%DB`→`202604明芝`）。要增刪路徑改這個檔即可。

## 2. 產出物
| 檔 | 內容 |
|---|---|
| `$SSD/<base>/listing.txt` | 該 base 遞迴列檔（每行一路徑） |
| `$SSD/<base>/index.csv` | `pat_dir, chart_no, status`（掃 `.pdb` 得病歷號） |
| `$SSD/<base>/pat_log.csv` | **每個 `.pat`：`病歷號 + .sdb 數`**（per-base） |
| `$SSD/pat_log_all.csv` | **合併總表**：`base, folder, pat_name, chart_no, n_sdb, n_pdb, n_files, matched` |
| `$SSD/cohort.txt` | 兩份 Excel 抽出的病歷號清單 |
| `$SSD/<base>/cohort_raw/` | 命中 cohort 的完整 `.pat`（含 `.sdb`） |
| `$SSD/h5_output/` | cohort 轉出的 study `.h5`（餵 forecast_c） |

`pat_log_all.csv` 就是你要的「每個 `.pat` log」；`matched=yes` = 該眼病歷號在治療 Excel 裡。
**重複回診的病人**：同一病歷號會有多個 `.pat`（多次掃描）各佔一列、且 `.sdb` 數越多代表回診/掃描越多
→ 用 `chart_no` 排序就能挑出回診多次者，正是本專案要的對象。

## 3. 各腳本（也可單獨用）
- `list_remote.sh BASE OUT HOST USER [CHARSET]` — `lftp find` 整棵樹 → `listing.txt`。
- `mirror_list.sh REL_LIST BASE OUT HOST USER [GLOB] [PAR] [CHARSET]` — 平行鏡像相對 `.pat` 清單；
  `GLOB='*.pdb'` 只抓 pdb；留空＝完整。可續跑、支援巢狀路徑。
- `scan_pdb.py --input <pdb根> --repo-root . --out index.csv` — 開 `.pdb` 取病歷號（不解影像）。
- `cohort_list_standalone.py out.txt 檔1.xlsx 檔2.xlsx` — 純標準庫，抽兩檔所有 7–8 位整數當病歷號。
- `filter_pats.py --index --cohort --out` — index ∩ cohort → 要完整下載的 `.pat`。
- `pat_log.py --listing [label=]listing.txt --index index.csv --cohort cohort.txt --out` — 產 per-`.pat` log。

> 舊的 `run_nas_pipeline.sh` / `download_*.sh` / `fetch_cohort_*.sh` 是**單一路徑**版，仍可用；
> 多路徑請用 `run_all_paths.sh`。

## 4. ⚠️ 兩份 Excel 的 join 欄位（已實際解析驗證 2026-07-01）
| 檔 / sheet | join 欄 | 格式 | 備註 |
|---|---|---|---|
| **EYLEA** `重新整理過` | **A `病歷號`** | 7–8 位（少數 6）醫院病歷號 | 乾淨數字、無連字號 |
| **Pool** `data collection` | **D `Chart no.`** | 7 位×529、8 位×258、6 位×33… | **也是**醫院病歷號 |

- 兩欄都是 7–8 位醫院病歷號，**正好對上 `.pdb` surname 解出的病歷號**（`cohort_list` 抓 7–8 位整數即可命中）。
- **⚠️ 與 `CLAUDE.md §7` 矛盾**：§7 說「case-pooling 的 Chart no. 是別的序號，勿用」，但*這個版本*
  (`20250811更新`) 的 `Chart no.`(D 欄) 就是真病歷號，且 `Name`(E 欄) 放的是**真實姓名**（未去識別化），
  與 §7 描述的舊去識別化匯出不同。本流程靠「與 `.pdb` 病歷號取交集」自我驗證，兩欄都當 key 是安全的。
  詳見 `memory/excel-join-keys-verified.md`。
- 6 位的少數列（EYLEA 6 筆、pool 33 筆）會被「7–8 位」過濾掉；若要納入，把 `cohort_list_standalone.py`
  的 `^\d{7,8}$` 放寬為 `^\d{6,8}$`（交集驗證仍會擋掉雜訊）。

## 5. 編碼 / 疑難
- NAS 檔名是 **Big5**；腳本用 `ftp:charset=big5 + file:charset=utf-8`，`nas_bases.txt` 用 UTF-8 實名。
- 自簽憑證：`ssl:verify-certificate no`（內部信任 NAS）。
- 某些 base（`eye3/oct`、`eye3/hra`）可能不是 `.pat` 樹 → 無 `.pat` 就不下載/掃描，但 `pat_log` 仍會
  逐資料夾記錄 `.sdb` 數，不漏算。
- 轉完確認無誤後，可刪 `$SSD/<base>/pdb_only` 與 `cohort_raw` 省空間（`listing.txt`/`pat_log.csv` 留著）。
