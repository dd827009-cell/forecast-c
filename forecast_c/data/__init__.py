"""forecast_c.data — 配對/縱向 dataset（torch）。

最小版 = 配對版（visit t → t+Δ）。提供:
  - 批次契約（batch dict 欄位）說明
  - DummyPairedDataset / dummy_loader: 隨機張量，供 train loop smoke test
  - build_dataloader: L40 接真資料的 stub（清楚報缺件）
"""
from .dataset import (PAIRED_BATCH_KEYS, DummyPairedDataset, dummy_loader,
                      collate_paired, build_dataloader)

__all__ = ["PAIRED_BATCH_KEYS", "DummyPairedDataset", "dummy_loader",
           "collate_paired", "build_dataloader"]
