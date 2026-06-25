"""配對 OCT dataset（visit t → t+Δ）+ dummy + L40 stub。

批次契約（餵 ForecastModel.forward）:
  v_t          : encoder 吃的「現在」volume（real=OCT volume；dummy=(N,D) latent）
  v_future     : encoder 吃的「未來」volume（target，同上）
  treatment    : dict{drug_ids (M,), numerics (M,numeric_in), event_mask (M,), is_naive ()} 或 None
  dt           : 正規化（年）回診間隔 純量
  baseline     : baseline 嚴重度（CST z-score；純量或 (baseline_dim,)）
  thickness_gt : (out_h, out_w) 未來厚度圖 µm（真 GT；無則省略 → 不算厚度損失）
  thickness_mask: (out_h, out_w) 有效遮罩（可選）

🔴 病人層級切分（防洩漏）由上游 split 保證：測試眼的縱向配對不得進訓練。
"""
import torch
from torch.utils.data import Dataset, DataLoader

PAIRED_BATCH_KEYS = ["v_t", "v_future", "treatment", "dt", "baseline", "thickness_gt"]


def collate_paired(samples):
    """把 list[sample dict] 疊成 batch；treatment 子 dict 逐鍵 stack；None 治療整批設 None。"""
    batch = {}
    for k in ("v_t", "v_future", "dt", "baseline"):
        batch[k] = torch.stack([s[k] for s in samples])
    if all(s.get("treatment") is not None for s in samples):
        keys = samples[0]["treatment"].keys()
        batch["treatment"] = {kk: torch.stack([s["treatment"][kk] for s in samples]) for kk in keys}
    else:
        batch["treatment"] = None
    if all("thickness_gt" in s for s in samples):
        batch["thickness_gt"] = torch.stack([s["thickness_gt"] for s in samples])
    return batch


class DummyPairedDataset(Dataset):
    """隨機張量配對 dataset（v_t/v_future = (N,D) latent，配 DummyEncoder）→ smoke test train loop。"""

    def __init__(self, cfg, n=64, with_treatment=True, with_thickness=True, m_events=3):
        self.cfg, self.n = cfg, n
        self.with_treatment, self.with_thickness, self.m = with_treatment, with_thickness, m_events

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        N, D = self.cfg.backbone.n_tokens, self.cfg.backbone.embed_dim
        s = {"v_t": torch.randn(N, D), "v_future": torch.randn(N, D),
             "dt": torch.rand(()), "baseline": torch.rand(())}
        if self.with_treatment:
            s["treatment"] = {
                "drug_ids": torch.randint(1, self.cfg.treat.n_drug_types, (self.m,)),
                "numerics": torch.rand(self.m, self.cfg.treat.numeric_in),
                "event_mask": torch.ones(self.m, dtype=torch.bool),
                "is_naive": torch.zeros((), dtype=torch.bool)}
        else:
            s["treatment"] = None
        if self.with_thickness:
            s["thickness_gt"] = torch.rand(self.cfg.thickness.out_h, self.cfg.thickness.out_w) * 400.0
        return s


def dummy_loader(cfg, batch_size=8, n=64, **kw):
    """smoke-test DataLoader（dummy 配對）。"""
    return DataLoader(DummyPairedDataset(cfg, n=n, **kw), batch_size=batch_size,
                      shuffle=True, collate_fn=collate_paired)


def build_dataloader(cfg, rank=0, world=1, split="train"):
    """L40 接點: 縱向 shard → 配對 batch（治療 + Δt + baseline + 厚度 GT）。

    需 (L40): 縱向 manifest（病人層級 split）、M7b 層（厚度 GT）、治療 metadata、OCTCube 吃的 volume。
    """
    raise NotImplementedError(
        "build_dataloader 是 L40 接點：需縱向配對 shard（病人層級 split 防洩漏）+ 治療 metadata + "
        "厚度 GT（M7b）+ OCTCube volume。本機 smoke test 請用 dummy_loader(cfg)。")


# ───────────────────────── dummy 自測（`python -m forecast_c.data.dataset`） ─────────────────────────
if __name__ == "__main__":
    from forecast_c.config import ForecastConfig
    cfg = ForecastConfig.tiny()
    loader = dummy_loader(cfg, batch_size=4, n=16)
    batch = next(iter(loader))
    N, D = cfg.backbone.n_tokens, cfg.backbone.embed_dim
    assert batch["v_t"].shape == (4, N, D) and batch["v_future"].shape == (4, N, D)
    assert batch["dt"].shape == (4,) and batch["baseline"].shape == (4,)
    assert batch["treatment"]["drug_ids"].shape == (4, 3)
    assert batch["thickness_gt"].shape == (4, cfg.thickness.out_h, cfg.thickness.out_w)
    # 治療可關: with_treatment=False → batch treatment=None
    loader2 = dummy_loader(cfg, batch_size=4, n=8, with_treatment=False)
    assert next(iter(loader2))["treatment"] is None
    print("dataset dummy 自測通過 ✅  batch keys:", list(batch.keys()))
