"""模塊 C: 治療編碼器。把「兩次 visit 間打什麼針」→ 向量 a（餵 predictor 條件注入）。

設計（完整規劃_C §5）:
  - 藥身份 embedding（主訊號）+ 數值 MLP（次數/距末次天數/累積）分開編碼後 concat → 投到 a。
    身份與數值分離 = 反事實鋪路（第二篇做換藥只抽換 drug_ids，數值不動）。
  - 多藥時間加權聚合（距末次天數越小 = 越近 → 權重越大）。
  - treatment-naive 專屬「無治療」embedding（對照組 / §7.2 治療可關）。
"""
import torch
import torch.nn as nn

from forecast_c.config import TreatmentEncoderConfig


class TreatmentEncoder(nn.Module):
    """forward(treatment) -> a  (B, out_dim)

    treatment dict:
      drug_ids   : (B, M) long      M 個用藥事件的藥種 id（0=padding/缺）
      numerics   : (B, M, numeric_in) float  每事件 [次數, 距末次天數, 累積]
      event_mask : (B, M) bool      哪些事件有效
      is_naive   : (B,) bool        treatment-naive → 走專屬 embedding（對照）
    """

    def __init__(self, cfg: TreatmentEncoderConfig):
        super().__init__()
        self.cfg = cfg
        de = cfg.drug_embed_dim
        self.drug_emb = nn.Embedding(cfg.n_drug_types, de, padding_idx=0)
        self.numeric_mlp = nn.Sequential(
            nn.Linear(cfg.numeric_in, cfg.numeric_hidden), nn.GELU(),
            nn.Linear(cfg.numeric_hidden, de))
        self.event_proj = nn.Linear(de * 2, cfg.out_dim)         # [身份, 數值] → a
        if cfg.use_naive_embedding:
            self.naive_emb = nn.Parameter(torch.zeros(cfg.out_dim))

    def forward(self, t: dict) -> torch.Tensor:
        drug_ids = t["drug_ids"]                     # (B,M)
        numerics = t["numerics"]                     # (B,M,numeric_in)
        mask = t["event_mask"].float()               # (B,M)

        d = self.drug_emb(drug_ids)                  # (B,M,de) 身份
        n = self.numeric_mlp(numerics)               # (B,M,de) 數值
        ev = self.event_proj(torch.cat([d, n], dim=-1))   # (B,M,out)

        # 時間加權聚合: 距末次天數 (numerics[...,1]) 越小 → 越近 → 權重越大
        days = numerics[..., 1]
        score = (-days).masked_fill(mask == 0, -1e9)
        w = torch.softmax(score, dim=1)
        w = torch.nan_to_num(w)                      # 全 padding 列保險
        agg = (ev * w.unsqueeze(-1)).sum(dim=1)      # (B,out)

        if self.cfg.use_naive_embedding:
            is_naive = t["is_naive"].view(-1, 1).float()
            return is_naive * self.naive_emb + (1.0 - is_naive) * agg
        return agg


# ───────────────────────── dummy 自測（`python -m forecast_c.model.treatment`） ─────────────────────────
if __name__ == "__main__":
    from forecast_c.config import ForecastConfig
    cfg = ForecastConfig.tiny()
    enc = TreatmentEncoder(cfg.treat)
    B, M = 4, 3
    treat = {
        "drug_ids": torch.randint(1, cfg.treat.n_drug_types, (B, M)),
        "numerics": torch.rand(B, M, cfg.treat.numeric_in),
        "event_mask": torch.ones(B, M, dtype=torch.bool),
        "is_naive": torch.tensor([True, False, False, False]),
    }
    a = enc(treat)
    assert a.shape == (B, cfg.treat.out_dim), a.shape
    a.sum().backward()
    assert any(p.grad is not None for p in enc.parameters()), "無梯度"
    # padding 事件不影響聚合（全 padding 列不 NaN）
    treat2 = dict(treat, event_mask=torch.zeros(B, M, dtype=torch.bool),
                  is_naive=torch.zeros(B, dtype=torch.bool))
    assert torch.isfinite(enc(treat2)).all(), "全 padding 應不 NaN"
    print("treatment dummy 自測通過 ✅")
