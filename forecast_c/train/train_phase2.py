"""Phase 2 訓練入口: 凍結 encoder + 一步 predictor + 厚度頭（時間 JEPA，★無 EMA）。

用法:
  smoke（本機/容器，dummy encoder + dummy loader，驗迴圈跑通）:
      python -m forecast_c.train.train_phase2 --smoke --steps 5
  真訓練（L40）:
      python -m forecast_c.train.train_phase2 --config configs/phase2.json
      ⚠️ build_encoder / build_dataloader 待接真 OCTCube + 縱向配對 shard（L40）。

與前幾代差異: **無 ema_step**（target 由同一凍結 encoder 編未來 stop-grad）；optimizer 只收
可訓練參數（predictor + treatment + thickness），encoder 凍結不進。
"""
import argparse

import torch

from forecast_c.config import ForecastConfig
from forecast_c.model import ForecastModel, ThicknessHead, DummyEncoder, build_octcube_encoder
from forecast_c.data import dummy_loader, build_dataloader


def build_encoder(cfg, ckpt=None):
    """L40 接點: 官方 OCTCube（凍結）。本機請走 --smoke（DummyEncoder）。"""
    return build_octcube_encoder(ckpt_path=ckpt)        # 目前 raise NotImplementedError（清楚報缺件）


def build_model(cfg, encoder):
    """ForecastModel = 凍結 encoder + predictor + 厚度頭。"""
    thickness = ThicknessHead(cfg.backbone, cfg.thickness)
    return ForecastModel(encoder, cfg, thickness_head=thickness)


def trainable_params(model):
    """只收 requires_grad 的參數（凍結 encoder 自動排除）。"""
    return [p for p in model.parameters() if p.requires_grad]


def step_fn(model, batch, device):
    g = lambda k: batch[k].to(device) if torch.is_tensor(batch.get(k)) else batch.get(k)
    treat = None
    if batch.get("treatment") is not None:
        treat = {k: v.to(device) for k, v in batch["treatment"].items()}
    out = model(g("v_t"), g("v_future"), treat, g("dt"), g("baseline"),
                thickness_gt=g("thickness_gt"))
    return out["loss"], out["detail"]


def train_loop(model, loader, cfg, device, steps=None, lr=1e-4, log_every=1):
    model.to(device)
    opt = torch.optim.AdamW(trainable_params(model), lr=lr)
    model.train()
    step = 0
    for batch in loader:
        loss, detail = step_fn(model, batch, device)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0:
            print(f"  step {step:4d}  loss={float(loss):.4f}  {detail}")
        step += 1
        if steps and step >= steps:
            break
    return step


def main():
    ap = argparse.ArgumentParser(description="Phase 2 訓練（設計 C 最小版）")
    ap.add_argument("--smoke", action="store_true", help="dummy encoder+loader 驗迴圈（不需權重/資料）")
    ap.add_argument("--config", help="真訓練 config（L40）")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if a.smoke:
        cfg = ForecastConfig.tiny()
        model = build_model(cfg, DummyEncoder(cfg.backbone.embed_dim))
        loader = dummy_loader(cfg, batch_size=a.batch_size, n=a.batch_size * a.steps)
        print(f"[smoke] device={device}  可訓練張量數={len(trainable_params(model))} "
              f"（encoder 凍結排除）")
        train_loop(model, loader, cfg, device, steps=a.steps, lr=a.lr)
        print("[smoke] 訓練迴圈跑通 ✅（★無 EMA、encoder 凍結）")
        return

    # 真訓練（L40）
    cfg = ForecastConfig()              # TODO(L40): 從 a.config 載
    encoder = build_encoder(cfg)        # NotImplementedError until L40
    model = build_model(cfg, encoder)
    loader = build_dataloader(cfg)      # NotImplementedError until L40
    train_loop(model, loader, cfg, device, steps=a.steps, lr=a.lr)


if __name__ == "__main__":
    main()
