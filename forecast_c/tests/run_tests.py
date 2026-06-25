"""forecast_c 整合測（dummy-latent）。

跑: docker run --rm -v "$(pwd)":/workspace -w /workspace octcube-dev \
        python -m forecast_c.tests.run_tests
框架: 未實作 → PENDING；實作 → PASS；壞 → FAIL（exit 1）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from forecast_c.config import ForecastConfig
from forecast_c.model import (FrozenEncoder, DummyEncoder, build_octcube_encoder,
                              TreatmentEncoder, OneStepPredictor, AdaLNZeroBlock,
                              FiLMZeroBlock, ThicknessHead, predict_loss, ForecastModel,
                              baselines)
from forecast_c.data import dummy_loader
from forecast_c.train import eval as ev
from forecast_c.train.train_phase2 import build_model, train_loop, trainable_params

CFG = ForecastConfig.tiny()


# ---------------------------- config ----------------------------
def test_config_derived():
    c = ForecastConfig.tiny()
    assert c.predictor.in_dim == c.backbone.embed_dim
    assert c.predictor.dt_dim == 2 * c.predictor.dt_fourier_bands
    assert c.predictor.cond_dim == c.treat.out_dim + c.predictor.dt_dim + c.predictor.baseline_dim


# ---------------------------- encoder (★ 凍結 + 免 EMA) ----------------------------
def test_encoder_frozen_and_eval():
    enc = FrozenEncoder(DummyEncoder(CFG.backbone.embed_dim))
    assert all(not p.requires_grad for p in enc.parameters())
    enc.train()                                   # 不該解凍 inner
    assert not enc.inner.training
    tok, cls = enc(torch.randn(2, CFG.backbone.n_tokens, CFG.backbone.embed_dim))
    assert tok.grad_fn is None                     # stop-grad


def test_octcube_remap_splits_qkv():
    """OCTCube flash→非flash remap: mixer.Wqkv(3072) 拆成 attn.q/k/v(各1024) + 丟 decoder/mask。"""
    from forecast_c.model.encoder import _octcube_remap
    sd = {"blocks.0.mixer.Wqkv.weight": torch.randn(3072, 1024),
          "blocks.0.mixer.Wqkv.bias": torch.randn(3072),
          "blocks.0.mixer.out_proj.weight": torch.randn(1024, 1024),
          "decoder_blocks.0.x": torch.randn(2), "mask_token": torch.randn(2),
          "norm.weight": torch.randn(1024)}
    out = _octcube_remap(sd)
    for s in ("q", "k", "v"):
        assert out[f"blocks.0.attn.{s}.weight"].shape == (1024, 1024)
        assert out[f"blocks.0.attn.{s}.bias"].shape == (1024,)
    assert "blocks.0.attn.proj.weight" in out
    assert "decoder_blocks.0.x" not in out and "mask_token" not in out and "norm.weight" in out


# ---------------------------- 模塊 C/D ----------------------------
def test_treatment_and_off():
    enc = TreatmentEncoder(CFG.treat)
    B, M = 4, 3
    t = {"drug_ids": torch.randint(1, CFG.treat.n_drug_types, (B, M)),
         "numerics": torch.rand(B, M, CFG.treat.numeric_in),
         "event_mask": torch.ones(B, M, dtype=torch.bool),
         "is_naive": torch.tensor([True, False, False, False])}
    a = enc(t)
    assert a.shape == (B, CFG.treat.out_dim)
    a.sum().backward()
    assert any(p.grad is not None for p in enc.parameters())


def test_adaln_film_identity():
    x = torch.randn(2, 16, CFG.predictor.width)
    cond = torch.randn(2, CFG.predictor.cond_dim)
    assert torch.allclose(AdaLNZeroBlock(CFG.predictor)(x, cond), x, atol=1e-5)
    assert torch.allclose(FiLMZeroBlock(CFG.predictor)(x, cond), x, atol=1e-5)


def test_predictor_residual_start():
    pred = OneStepPredictor(CFG.predictor)
    z_t = torch.randn(2, CFG.backbone.n_tokens, CFG.predictor.in_dim)
    z_hat, logvar = pred(z_t, torch.randn(2, CFG.predictor.cond_dim))
    assert z_hat.shape == z_t.shape and logvar.shape == (2, CFG.backbone.n_tokens, 1)
    assert torch.allclose(z_hat, z_t, atol=1e-6), "殘差零初始 → 起點 ẑ=z_t"


# ---------------------------- losses ----------------------------
def test_losses():
    z = torch.randn(2, 32, 16)
    assert float(predict_loss(z, z.clone(), cfg=CFG.loss)) < 1e-6
    z_hat = torch.randn(2, 16, 8); z_tgt = z_hat + 1.0
    lo = predict_loss(z_hat, z_tgt, logvar=torch.full((2, 16, 1), -2.0), cfg=CFG.loss)
    hi = predict_loss(z_hat, z_tgt, logvar=torch.full((2, 16, 1), 2.0), cfg=CFG.loss)
    assert float(hi) < float(lo)                   # NLL 異方差
    z0 = torch.zeros(1, 8, 4)
    l = float(predict_loss(torch.randn(1, 8, 4), z0.clone(), z_t=z0, cfg=CFG.loss))
    assert l == l and 0.0 < l < 1e6                # change-weight floor


# ---------------------------- thickness ----------------------------
def test_thickness_shape():
    head = ThicknessHead(CFG.backbone, CFG.thickness)
    out = head(torch.randn(2, CFG.backbone.n_tokens, CFG.backbone.embed_dim))
    assert out.shape == (2, CFG.thickness.out_h, CFG.thickness.out_w)


# ---------------------------- 整機 ForecastModel ----------------------------
def _model():
    th = ThicknessHead(CFG.backbone, CFG.thickness)
    return ForecastModel(DummyEncoder(CFG.backbone.embed_dim), CFG, thickness_head=th)


def _treat(B):
    return {"drug_ids": torch.randint(1, CFG.treat.n_drug_types, (B, 3)),
            "numerics": torch.rand(B, 3, CFG.treat.numeric_in),
            "event_mask": torch.ones(B, 3, dtype=torch.bool),
            "is_naive": torch.zeros(B, dtype=torch.bool)}


def test_forecast_end_to_end_and_no_ema():
    m = _model()
    assert not hasattr(m, "teacher"), "★ 最小版免 EMA（不該有 teacher）"
    B, N, D = 2, CFG.backbone.n_tokens, CFG.backbone.embed_dim
    vt, vf = torch.randn(B, N, D), torch.randn(B, N, D)
    gt = torch.rand(B, CFG.thickness.out_h, CFG.thickness.out_w) * 400
    out = m(vt, vf, _treat(B), torch.rand(B), torch.rand(B), thickness_gt=gt)
    assert out["z_hat"].shape == (B, N, D) and "thickness" in out["detail"]
    out["loss"].backward()
    assert all(p.grad is None for p in m.encoder.parameters()), "凍結 encoder 不該有梯度"
    assert m.predictor.out_proj.weight.grad is not None
    assert m.thickness_head.proj.weight.grad is not None


def test_forecast_treatment_off_and_persistence_start():
    m = _model()
    B, N, D = 2, CFG.backbone.n_tokens, CFG.backbone.embed_dim
    vt, vf = torch.randn(B, N, D), torch.randn(B, N, D)
    out = m(vt, vf, None, torch.rand(B), torch.rand(B))       # 治療可關
    assert out["z_hat"].shape == (B, N, D)
    v = torch.randn(B, N, D)
    out2 = m(v, v, None, torch.rand(B), torch.rand(B))        # future==present
    assert out2["detail"]["predict"] < 1e-4, out2["detail"]["predict"]


# ---------------------------- baselines + eval ----------------------------
def test_baselines():
    B, N, D = 2, CFG.backbone.n_tokens, CFG.backbone.embed_dim
    z_t = torch.randn(B, N, D)
    assert torch.equal(baselines.persistence(z_t), z_t)
    mc = baselines.MeanChangeBaseline(D); assert torch.allclose(mc(z_t), z_t)
    dr = baselines.DirectThicknessRegressor(CFG.backbone, CFG.thickness, CFG.predictor.cond_dim)
    out = dr(z_t, torch.randn(B, CFG.predictor.cond_dim))
    assert out.shape == (B, CFG.thickness.out_h, CFG.thickness.out_w)


def test_eval_metrics():
    B, N, D = 32, 16, 8
    z_t = torch.randn(B, N, D); delta = torch.randn(B, N, D) * 0.5
    z_tgt = z_t + delta
    assert ev.persistence_skill(z_tgt.clone(), z_tgt, z_t)["skill"] > 0.99   # 完美
    assert abs(ev.persistence_skill(z_t.clone(), z_tgt, z_t)["skill"]) < 1e-5  # persistence
    gt = torch.rand(B, 5, 8) * 400
    cmp = ev.compare_to_direct(gt + torch.randn_like(gt) * 2, gt + torch.randn_like(gt) * 50, gt)
    assert cmp["latent_wins"]
    assert len(ev.change_conditioned(z_t + 0.5 * delta, z_tgt, z_t, n_bins=3)) >= 2


# ---------------------------- data + train smoke ----------------------------
def test_train_smoke():
    m = build_model(CFG, DummyEncoder(CFG.backbone.embed_dim))
    loader = dummy_loader(CFG, batch_size=4, n=8)
    assert len(trainable_params(m)) > 0
    n = train_loop(m, loader, CFG, torch.device("cpu"), steps=2, log_every=99)
    assert n == 2


# ---------------------------- phase1 (已實作: vendored 前輩 code + 原創 JEPA) ----------------------------
def test_phase1_adapter3d_and_jepa():
    import torch.nn as nn2
    from forecast_c.phase1 import Adapter3D, SpatialJEPA, slice_mask, TeacherFeatureExtractor
    dim, d_size, hw = 32, 4, 4
    x = torch.randn(2 * d_size, hw * hw + 1, dim)
    # Adapter3D 起點恆等（殘差零初始）
    ad = Adapter3D(dim, adapter_channels=8, d_size=d_size, has_cls=True)
    assert torch.allclose(ad(x), x, atol=1e-6), "Adapter3D 起點應恆等"
    # 空間 JEPA（原創）: 只訓 adapter，base 凍結
    base = nn2.ModuleList([nn2.Linear(dim, dim) for _ in range(2)])
    jepa = SpatialJEPA(base, dim, d_size, adapter_channels=8, mask_frac=0.5)
    out = jepa(x)
    assert torch.isfinite(out["loss"]) and 0 < out["n_masked"] < 2 * d_size
    out["loss"].backward()
    assert all(p.grad is None for p in jepa.blocks.parameters()), "JEPA base 凍結"
    assert jepa.mask_token.grad is not None
    # 教師 extractor: MedSAM2 仍待接 sam2（RETFound/DINOv3 已可從本地載入，見 teachers.py self-test）
    try:
        TeacherFeatureExtractor(ckpt_dir="ckpts").load("medsam2")
    except (NotImplementedError, FileNotFoundError):
        pass
    else:
        raise AssertionError("MedSAM2 extractor 應待接 sam2")


# ---------------------------- phase3 stub ----------------------------
def test_phase3_stub_raises():
    from forecast_c.phase3 import SwinUNETRDecoder, fluid_volume_change
    for fn in [lambda: SwinUNETRDecoder((8, 8, 8), 1, 2).build(),
               lambda: fluid_volume_change(None, None)]:
        try:
            fn()
        except NotImplementedError:
            continue
        raise AssertionError("phase3 應為 stub（NotImplementedError）")


TESTS = [
    ("config 衍生維度", test_config_derived),
    ("★ encoder 凍結 + 恆 eval", test_encoder_frozen_and_eval),
    ("OCTCube flash→非flash remap(Wqkv拆qkv)", test_octcube_remap_splits_qkv),
    ("C 治療編碼器 + 治療可關", test_treatment_and_off),
    ("D AdaLN/FiLM 起點恆等", test_adaln_film_identity),
    ("D predictor 殘差起點", test_predictor_residual_start),
    ("L predict/NLL/change-floor", test_losses),
    ("B 厚度頭形狀", test_thickness_shape),
    ("整機 端到端 + ★免EMA + 凍結無梯度", test_forecast_end_to_end_and_no_ema),
    ("整機 治療可關 + persistence 起點≈0", test_forecast_treatment_off_and_persistence_start),
    ("baselines（persistence/mean/direct）", test_baselines),
    ("eval（P0 skill / L 直接回歸 / change-cond）", test_eval_metrics),
    ("data + train smoke（無 EMA、凍結）", test_train_smoke),
    ("phase1 Adapter3D起點恆等 + 空間JEPA(原創)", test_phase1_adapter3d_and_jepa),
    ("phase3 stub raise", test_phase3_stub_raises),
]


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 60)
    print("forecast_c 整合測（最小版）")
    print("=" * 60)
    n_pass = n_pend = n_fail = 0
    for name, fn in TESTS:
        try:
            fn(); print(f"  [PASS]    {name}"); n_pass += 1
        except NotImplementedError:
            print(f"  [PENDING] {name}"); n_pend += 1
        except Exception as e:
            print(f"  [FAIL]    {name}  -> {type(e).__name__}: {e}"); n_fail += 1
    print("-" * 60)
    print(f"PASS {n_pass} / PENDING {n_pend} / FAIL {n_fail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
