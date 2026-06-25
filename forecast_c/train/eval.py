"""存亡評估（完整規劃_C §8 / 待辦 §存亡兩層）。

兩條鐵律:
  P0 — 會變的眼上**贏得了 persistence 嗎**？贏不了 → 任務不可測 → 退 Phase 1 FM。
  L  — latent 預測**贏得了「直接回歸」嗎**？輸了 → 方法無理由 → 重新定位。
另: change-conditioned（按變化量分層的誤差），證明非只在不變的眼上虛胖。

純指標計算（吃預測/目標張量）→ 可用合成數值驗邏輯，真資料時餵真預測。
"""
import torch


def _err(pred, target, reduce_dims=(-1,)):
    """逐 sample 誤差（L2，over reduce_dims 後再對其餘非 batch 維平均）。pred/target: (B, ...)。"""
    e = (pred - target).pow(2).sum(dim=reduce_dims).sqrt()       # (B, ...) 去掉 reduce_dims
    while e.dim() > 1:
        e = e.mean(dim=-1)
    return e                                                     # (B,)


def changing_mask(z_t, z_target, frac_thresh=0.5):
    """會變的眼遮罩: ‖z_target−z_t‖ 大於 batch 中位數 × ... → 這裡用「> 中位數」當「會變」。

    回傳 (B,) bool。frac_thresh 為分位（0.5=中位數以上算會變）。
    """
    chg = _err(z_target, z_t)                                    # (B,)
    thr = torch.quantile(chg, frac_thresh)
    return chg > thr


def persistence_skill(z_hat, z_target, z_t, on_changing=True):
    """P0: persistence skill = 1 − err(model)/err(persistence)。>0 = 贏 persistence。

    persistence 預測 = z_t（copy-last）。on_changing=True → 只在會變的眼上算（鐵律母體）。
    回傳 dict{skill, err_model, err_persist, n}。
    """
    err_model = _err(z_hat, z_target)
    err_persist = _err(z_t, z_target)
    if on_changing:
        m = changing_mask(z_t, z_target)
        if m.sum() == 0:
            m = torch.ones_like(m)
        err_model, err_persist = err_model[m], err_persist[m]
    em, ep = float(err_model.mean()), float(err_persist.mean())
    return {"skill": 1.0 - em / (ep + 1e-9), "err_model": em, "err_persist": ep,
            "n": int(err_model.numel())}


def compare_to_direct(thick_latent, thick_direct, thick_gt):
    """L: 比 latent 預測解碼厚度 vs 直接回歸厚度（對真 GT 的 µm 誤差）。

    回傳 dict{err_latent, err_direct, latent_wins, rel_improve}。latent_wins=True → 方法有理由。
    """
    el = float(_err(thick_latent, thick_gt).mean())
    ed = float(_err(thick_direct, thick_gt).mean())
    return {"err_latent": el, "err_direct": ed, "latent_wins": el < ed,
            "rel_improve": (ed - el) / (ed + 1e-9)}


def change_conditioned(z_hat, z_target, z_t, n_bins=3):
    """按變化量分層的模型/persistence 誤差（證明非只在不變的眼上虛胖）。

    回傳 list[{bin, lo, hi, n, err_model, err_persist, skill}]（變化量由小到大）。
    """
    chg = _err(z_target, z_t)
    em = _err(z_hat, z_target)
    ep = _err(z_t, z_target)
    edges = torch.quantile(chg, torch.linspace(0, 1, n_bins + 1))
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (chg >= lo) & (chg <= hi if b == n_bins - 1 else chg < hi)
        if m.sum() == 0:
            continue
        e_m, e_p = float(em[m].mean()), float(ep[m].mean())
        rows.append({"bin": b, "lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "err_model": e_m, "err_persist": e_p, "skill": 1.0 - e_m / (e_p + 1e-9)})
    return rows


# ───────────────────────── dummy 自測（`python -m forecast_c.train.eval`） ─────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, D = 64, 32, 16
    z_t = torch.randn(B, N, D)
    delta = torch.randn(B, N, D) * 0.5
    z_target = z_t + delta

    # 完美模型（z_hat=z_target）→ skill≈1（贏 persistence）
    perfect = persistence_skill(z_target.clone(), z_target, z_t)
    assert perfect["skill"] > 0.99, perfect

    # persistence 模型（z_hat=z_t）→ skill≈0（沒贏）
    none_skill = persistence_skill(z_t.clone(), z_target, z_t)
    assert abs(none_skill["skill"]) < 1e-5, none_skill

    # 部分學會（z_hat=z_t+0.5Δ）→ 0<skill<1
    half = persistence_skill(z_t + 0.5 * delta, z_target, z_t)
    assert 0.0 < half["skill"] < 1.0, half

    # L 對照: latent 厚度更準 → latent_wins
    gt = torch.rand(B, 5, 8) * 400
    cmp = compare_to_direct(gt + torch.randn_like(gt) * 2, gt + torch.randn_like(gt) * 50, gt)
    assert cmp["latent_wins"] and cmp["rel_improve"] > 0, cmp

    # change-conditioned: 分層、單調 bin 邊界
    rows = change_conditioned(z_t + 0.5 * delta, z_target, z_t, n_bins=3)
    assert len(rows) >= 2 and rows[0]["lo"] <= rows[-1]["hi"]
    print("eval dummy 自測通過 ✅  (perfect skill=%.2f, half=%.2f)" % (perfect["skill"], half["skill"]))
