---
name: optional-directions
description: "兩個使用者要求記錄的可選方向(未拍板):I-1 治療反應分型(latent軌跡聚類,建議優先) + I-2 擴散模型(只當質化影像渲染器,別當頭條)。詳見 solutions/TODO_待辦.md §I。"
metadata: 
  node_type: memory
  type: project
  originSessionId: a552fdf0-9f5c-4e6b-85fa-6bf05c68f6d2
---

使用者 2026-06-16 問「時序+聚類」與「擴散模型」是否好方向，**要求當可選項記錄**（未拍板）。已寫進 `solutions/TODO_待辦.md` §I。屬於 [[stage-a-redesign]] 的延伸選項。

- **I-1 聚類（建議優先）**：凍結 latent 軌跡聚類→治療反應分型(快/慢/無反應/復發)。**不動 backbone、免算力**，當**下游分析章節(第二貢獻)**非新支線。驗證必綁臨床外變數：KM+log-rank(接存活頭閉環)、用藥分布、bootstrap 穩定性。先決=要真縱向資料。
- **I-2 擴散（窄範圍可選）**：最多當「未來影像渲染器」做質化圖(須過 decoding ceiling)；**別當頭條**——那是 EyeWorld 主場(生成式/OmniGen2/大資料)，比影像品質會輸。當 latent 機率預測器→丟 ablation(會弄糊 persistence baseline；多峰其實存活頭已抓)。

**紅線（使用者自己定的）**：別又變成「堆模組」。聚類=一個分析章節；擴散=換掉 optional 影像頭，非第三條支線。
