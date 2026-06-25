# 設計 C — 完整規劃(治療條件化 OCT Latent 預測)

> 對話收斂定版。取代設計 A(`_archive/A_StageA/`)、B(`_archive/B_OCTCube/`)。共用資產在 `solutions/`。
> 定位:**治療條件化 latent forecasting**,預測治療後**厚度 + 積水(體積變化)**。場子 **MICCAI / MIA**。
> **backbone 路線 = 方案①(2D 蒸餾 + 3D 聚合),解掉「2D 教師洗掉 3D」與「2D→3D lifting」兩個麻煩。**
> ⚠️ **用的是 JEPA「方法」,不是 V-JEPA「模型」**(見 §0.2)。**V-JEPA 模型在 ① 不使用**,僅「① 3D 太弱 / reviewer 要 joint-3D 對照」時的後備(②)。

---

## 0. 一句話定位
凍結的 OCT 表徵上,給「現在 + 治療 + Δt + 病人資訊」預測「未來 latent」,解碼成厚度µm + 積水體積變化;stop-grad target + persistence 檢查防作弊。

## 0.1 與前輩的關係(誠實標示)
- **2D 學生(多教師蒸餾 + MIM)≈ 楊瀚博**(教師換領域)。
- **3D 聚合 = 借吳韋論 3D adapter「概念」**(但用在 SSL,非他的雙分支分類)。
- **🆕 新(兩者都沒有)**:JEPA 訓 3D 聚合(空間 JEPA)+ 治療條件時間預測(時間 JEPA)。
- 性質:**整合級貢獻**(MICCAI/MIA);新穎度在「組合 + JEPA-3D + 治療預測」。

## 0.2 名詞澄清:JEPA(方法)≠ V-JEPA(模型)
- **JEPA = 訓練方法/配方**(遮 → 預測被遮區的表徵 + stop-grad target)。**本設計用它兩次**:①b 空間 JEPA(遮切片)、Phase 2 時間 JEPA(遮未來 visit)。
- **V-JEPA = 一個現成預訓練「影片模型」**(它當初用 JEPA 方法訓的)。**本設計(①)不使用 V-JEPA 模型**——3D 聚合是「你自己的模組、從零訓、用 JEPA 方法」。
- → 設計裡**沒有 V-JEPA 模型,只有 JEPA 方法 + 2D 蒸餾學生 + 3D 聚合模組**。

---

## 1. 資料
| 項目 | 內容 |
|---|---|
| 病人/眼 | ~600+ / ~1200 眼 |
| 回診 | 3-5 次 |
| 大語料 | ~20TB,每張有 ILM/BM(厚度真 GT)、同機型 |
| 積水 | 無 in-domain 標註 → MedSAM2 蒸餾 + 自標校準 |
| 治療 | 藥身份 + 注射次數(無劑量)|
| 病人 | 年齡、性別、Δt |
| 切分 | 病人層級(左右眼相關)|

**待普查 A-1**(`solutions/A1_census_spec.md`):眼數/轉移對/積水消退事件/治療分布/次數-時間 r/左右眼。

---

## 2. 架構總覽(方案①)

```
Phase 1 — 表徵(兩段)
  ①a 2D per-slice 學生:多教師蒸餾(DINOv2+RETFound+MedSAM2,per-teacher heads,2D→2D)+ 2D MIM → base 凍結
  ①b 跨切片 = 3D adapter 插進凍結學生(PEFT,base凍+residual;A1 拍板走 B)
       訓練(只訓 adapter):空間 JEPA(遮切片→預測被遮切片凍結特徵,stop-grad)
            + ILM/BM 厚度監督 + 時間相干性(縱向夠才加,見 §4)
       → 凍結(整個 encoder = 2D學生 + adapter)
        ↓
Phase 2 — 治療條件預測(時間 JEPA)
  z_t(token-wise)+(藥身份, 次數, Δt, 年齡, 性別)
   → predictor(殘差 + 單尺度治療條件 + change-weighted)→ ẑ_{t+Δ}
   target = 凍結 encoder 編未來(stop-grad,免 EMA)
   一步 + 短程 rollout(2-3 步)
        ↓
Phase 3 — 密集讀出
  3D 特徵 → SwinUNETR 解碼器 → 厚度µm + 積水分割→體積變化
  診斷:persistence skill / change-conditioned / decoding ceiling
```

**JEPA 在兩個尺度**:①b 空間 JEPA(遮切片=跨切片結構)/ Phase 2 時間 JEPA(遮未來 visit=跨回診動態)。

---

## 3. Phase 1a — 2D per-slice 學生

- **多教師蒸餾(2D→2D,乾淨、免 lifting、洗不掉)**:DINOv2(密集)+ RETFound(疾病)+ MedSAM2(積水),**per-teacher heads + 分布平衡**(不用 cross-attn 融合教師)。
- **2D MIM**:遮 patch → cross-attn decoder 補 → 對齊教師特徵(楊瀚博式)。
- ~~2.5D 鄰窗~~ **已被 B 取代**:跨切片改由 ①b 的 3D adapter **全程(每層)**處理,比 2.5D 局部更深 → 2D 學生保持**純 per-slice 蒸餾**(最乾淨)。2.5D 不再需要。
- 不用 CLIP(密集弱、無文字下游)。
- **訓完凍結** → 2D 知識 100% 保留,後續不動它。
- ⚠️ 工程:教師特徵**預計算**(per-B-scan × 3 教師,儲存量大,見 §11 問題)。

## 4. Phase 1b — 跨切片 = 3D adapter 插進凍結 2D 學生(A1 拍板走 B)

- **做法**:在凍結 2D 學生的 transformer blocks 插入輕量 **3D adapter**(depth conv + bottleneck,residual 起點恆等);**base 凍結、只訓 adapter**(PEFT,吳韋論驗過)→ 學生變跨切片感知,**穿透每層(深,勝 2.5D 局部/獨立聚合末端淺融合)**,且 residual **不洗掉**。
- **全方面勝獨立聚合**:跨切片更深 + PEFT 省參 + 單一機制 + 有驗證(理由見對話 Q1)。
- **空間 JEPA 訓 adapter**:遮 k 張切片 → 帶 adapter 的學生**預測被遮切片的表徵**;**target = 凍結學生對被遮切片的特徵**(stop-grad,穩定靶、不崩、免 EMA)。
- **⚠️ conv 必用 depth-only(3×1×1)**:只沿深度混、in-slice 1×1 → **保住切片內(in-slice)空間特徵不被擾動**;**別用 3×3×3**(會動到 in-slice)。in-slice 由「凍結 base + depth-only + residual」三重保護;訓 adapter 只動「跨切片/深度」軸,不動切片內。
- **adapter on/off = 天然消融**:off=純 2D 特徵、on=跨切片感知 → 直接驗「跨切片/空間 JEPA 有沒有用(K)」與「in-slice 有沒有被影響」。
- **+ ILM/BM 厚度監督**(3D 結構真幾何)。
- **+ 時間相干性(縱向子集)— 縱向夠才加(A-1 後定,非無條件)**:Δt 預測 / 雜訊不變性 → 讓 latent 對「**跨回診變化**」可預測(補 forecastability 缺口 C)。**前置 REG-1。**
  - ⚠️ **資料閘控**:縱向少到加不了 → forecastability 補不了 → Phase 2 預測恐弱(易塌 persistence/輸直接回歸);且縱向不足代表 Phase 2 本身也餓 → **退回 Phase 1 FM 論文(保底)**。
  - 🔴 **洩漏防範**:**病人層級切分必須同時套到此處**——測試眼的縱向配對**不能**進 Phase 1 時間相干性訓練(否則 encoder 已見過測試眼時間結構 → Phase 2 評估洩漏)。**很容易漏。**
- **崩塌監測**(std/RankMe/change-conditioned)當保險。
- **🚪 凍結前閘門(問題 7)**:Track A 凍結探測 **vs 現成 OCTCube**(厚度µm / 病理 AUC)——**贏 OCTCube 才凍結**(不是贏「現成 V-JEPA」,那無意義)。贏不了 → 重新定位敘事(強調多任務/縱向特化)。
- → 凍結整個 encoder。

## 5. Phase 2 — 治療條件預測

- 凍結 encoder → z_t(**token-wise,別 pool**)。
- predictor:`z_t +(藥身份 embed, 次數, Δt, 年齡, 性別)→ ẑ`
  - 殘差零初始(起點=persistence)
  - token-wise 空間 attention + **單尺度治療條件**(AdaLN/cross-attn,吃 encoder「2D學生+adapter」的 token)。**不宣稱「多尺度治療條件」**(predictor 沒有多尺度特徵);多尺度只在 Phase 3 SwinUNETR 讀出。除非聚合模組做成階層式才談多尺度條件(問題 6)。
  - change-weighted loss(抗 persistence)
- target = 凍結 encoder 編實際未來(stop-grad)= **時間 JEPA**,免 EMA。
- 一步 + 短程 rollout(2-3 步,scheduled sampling);不上長序列 SSM。
- **治療/時間共線**:預測任務兩個都餵不用解;歸因才靠真實間隔變異(A-1 算 r)。

## 6. Phase 3 — 密集讀出 + 積水體積

- 3D 特徵 → **SwinUNETR 多尺度 3D 解碼器** → 厚度圖 µm + 積水分割。
- **積水體積變化(Q3)**:每次回診分割→**真實 25 切片 slab 加總 @512 高解析**→體積;**這次 − 上次**;系統取樣誤差在變化量抵消。
  - 前提:**REG-1 配準(§7)** + **真實 25 切片** + **對的 spacing**。
  - 次要:ETDRS 中央子場體積(穩健);中心張只當輔助不當主體積。
- 積水知識:MedSAM2 蒸餾(在 2D 學生)+ ILM/BM 限視網膜帶內 + **自標 50-100 張校準**(誤差 + decoding ceiling)。
- **厚度=主(真 GT)、積水=次(pseudo-label)**。
- ⚠️ 背景壓制可能吃淺積水 → 校準確認對比。

## 7. REG-1 配準(用 ascan_pos_ir)
① `rpe_bm_y` 沿 RPE 拉平(去軸向)② baseline IR ↔ 回診 IR 2D 配準(血管)③ `ascan_pos_ir` 傳位移到 B-scan。**QC 閘門**:沒變區域殘差→0,沒過不進變化訊號。積水體積差**強需要**。骨架 `stage1/registration.py`。

## 8. 評估(`solutions/EVALUATION.md` + 補強)
- baseline:persistence/copy-last + treatment-blind + mean-change + Δt-only + **直接回歸**。
- 指標:厚度µm/積水mm³(對齊 decoding ceiling)、**persistence skill**、**change-conditioned**、校準 CRPS/ECE、rollout 退化。
- 統計:McNemar/DeLong/Wilcoxon + ≥3 seeds + bootstrap CI;**病人層級切分(🔴 同時套 Phase 1 時間相干性,測試眼不進 → 防洩漏)**。
- 分層:治療/嚴重度/病程/變化量。外部:OLIVES/MARIO/RETOUCH。
- **積水準度落在自標真 GT 子集(解循環 D)**:訓練用 MedSAM2 假標(沒得選),但**報準度時對「人標 50-100 張真 GT」**,不對假標報(否則只是量「像不像分割器」)。順便量分割器自身誤差 = 積水 decoding ceiling。

## 9. Phase 0 前置(不卡算力)
A-1 普查 · 自標積水校準 · R-1 約醫師(積水≠厚度,有積液標註優先)· related-work go/no-go(2D→3D / JEPA 聚合 / 治療 OCT 預測)。

## 10. 鎖定 / 待拍板
✅:方案①(2D 蒸餾 + **跨切片走 B:3D adapter 插進凍結學生**)· 教師 MedSAM2/RETFound/DINOv2 · 空間JEPA(訓adapter)+ 時間JEPA · SwinUNETR 解碼 · 病人層級切分 · 積水真分割→體積變化。
**多尺度來源**:要原生多尺度→階層式學生+ScaleKD 對齊;否則 plain ViT+ViT-Adapter;最小版簡單解碼器。
⬜:存活頭(視 A-1 事件)· 頻率/小波(積水弱才加)。
**② V-JEPA 模型:不預設做**,僅「① 3D 太弱 / reviewer 要 joint-3D 對照」時的後備(commit ①、不自相矛盾)。

---

## 11. 已知問題與修法(重新分析後)
> 見對話「重新分析」一節;摘要:
- **A 3D 從零學**(無預訓練 3D)→ 聚合模組夠不夠強?修:20TB 對「聚合模組」夠;+ 2.5D 補早期;② V-JEPA 當**後備**(非預設)。
- **B 晚融合**(2D per-slice 各自編碼)→ 3D 可能弱。**修:2.5D 鄰窗(±1~2)補早期跨切片(§3),長距離留聚合模組。** ✅ 已納入。
- **C forecastability**:空間 JEPA(跨切片)≠ 時間可預測(跨回診)。**修:Phase 1b 加時間相干性(縱向)。** ✅ 已改必加。
- **D 積水循環**:訓+評估同一 pseudo-label 分割器 → 評估落自標真 GT。
- E 凍結 2D 不能對 OCT co-adapt(保留 vs 適配張力)→ 可在 1b 對 2D 極低 LR 輕微 fine-tune(監測別洗掉)。
- F 空間 JEPA 受切片稀疏(240µm)限制 → 接受(資料固有)+ 厚度監督補。
- G 教師特徵預計算/儲存(5M B-scan × 3 教師)→ 子集預計算 / on-the-fly;先算存得下嗎。
- **H 元件太多 = 整合風險(問題 8)→ 先做最小可跑版**:2D 學生(先單教師也行)凍結 + 一步預測 + 厚度頭 → **出第一個數字**,再逐件加(2.5D / 3D 聚合 / 第二三教師 / 積水 / REG-1 / rollout),**每件過消融閘門**。
- **I REG-1 QC 吃資料(問題 9)**:配準沒過的 study 不進變化訊號 → **A-1 普查順便估「配準 QC 通過率」**(縮多少)。
- **J 抗作弊依賴乾淨配準+分割(問題 10)**:persistence/change-conditioned/只看會變眼 全靠「知道哪裡真的變」→ **先驗配準/分割品質(QC + 自標校準)再信抗作弊指標**。
- (問題 5 npj=分類:① 不用 V-JEPA 模型 → 已不相關,僅 ② 後備才需注意。)

### 最後嚴謹檢查新增(Phase 2/3 風險疊加)
- **K 空間 JEPA 可能 trivial 內插**(切片稀疏)→ 真 3D 可能靠厚度監督。**修:消融「空間 JEPA on/off」,沒比厚度-only 好就砍。**
- **L 直接回歸可能贏 latent 預測**(存亡風險)→ **修:「latent vs 直接回歸」當核心消融;輸了就重新定位**(賣標籤效率/多步/遷移,非單純準度)。
- **M 積水體積變化需分割器「跨回診一致」**(非只準度)→ **修:驗 test-retest 一致性。**
- **N off-manifold 解碼**:頭用真實 latent 訓、卻解碼預測 ẑ → **修:監測 ẑ 分布 vs 真實;殘差+stopgrad 貼近;必要時頭也在預測 latent 上訓。**
- **O SwinUNETR 解碼器要多尺度,來源** → **修:要原生多尺度=「階層式 2D 學生 + ScaleKD 式逐階段對齊頭」(蒸餾 plain-ViT 教師有標準對齊法,NeurIPS 2024);plain ViT + ViT-Adapter 當備選;最小版用簡單解碼器。多尺度 on/off 消融。adapter 是接線非貢獻。**
- 次要:3D 聚合須 token-preserving(別 pool);1a→1b 間加品質檢查;「2D→2D」措辭因 2.5D 應改「2.5D學生←2D中心張目標」。
- **P 洩漏(🔴 易漏)**:Phase 1 時間相干性用縱向配對 → **病人切分必須同時套到它**(測試眼不進),否則 encoder 見過測試眼時間結構 → Phase 2 評估洩漏。
- **A1 跨切片機制 ✅ 已拍板走 B(3D adapter 插進凍結學生)**:全方面勝(穿透每層更深 + PEFT 省參 + 單一機制 + 吳韋論驗過 + 不洗掉);2.5D + 獨立聚合已棄。

### 存亡兩層(最小版先驗,見待辦)
- **P0(任務可不可測)**:會變的眼上贏得了 persistence 嗎?贏不了 → 預測招牌不成立 → 退 Phase 1 FM。
- **L(方法值不值得)**:latent 預測贏得了「直接回歸」嗎?輸了 → 方法無理由 → 重新定位。
- (K 空間 JEPA trivial = 元件值不值得,非存亡。)

---

## 12. 共用資產(`solutions/`)
前處理 Pipeline · 普查 A1 · 評估 EVALUATION · 文獻 RELATED_WORK · RUNBOOK。
程式:stage0/1 · multitask(厚度頭)· latent_dynamics(predictor/losses/registration/survival)· backbone(ViT-Adapter/3D adapter)· peft · train。

## 一頁總結
> **Phase 1a**:DINOv2+RETFound+MedSAM2 蒸餾出 2D 學生(2D→2D 乾淨)+2D MIM → 凍結。**Phase 1b**:3D 聚合模組(跨切片)用空間 JEPA(遮切片→預測凍結 2D 特徵)+厚度監督訓 → 凍結。**Phase 2**:治療條件時間 JEPA 預測。**Phase 3**:SwinUNETR 解碼 → 厚度+積水體積變化(真25切片@512+配準)。**先做**:A-1+自標+約醫師+文獻閘門。
