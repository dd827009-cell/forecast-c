"""變乾（recovery）標籤 — 從 CST 軌跡推導 (event_step, event_observed)。

- 變乾定義 = CST < 乾閾值（分病種閾值待醫師定，A-1 普查先用暫定值如 300µm）。
- 右設限 = 整條軌跡到最後都沒變乾（還在治療 / 失訪）。
- NaN CST（層分割失敗）→ 視為「未變乾」（不誤判）。

A-1 普查用「①變乾事件數」決定要不要做存活頭。純 numpy / stdlib（無 torch）。
self-test 見檔尾。
"""
import numpy as np


def is_dry(cst_um, threshold_um) -> bool:
    """CST（µm）→ 是否變乾（CST < 閾值）。NaN（無效層）→ False（視為未變乾，不誤判）。"""
    return bool(np.isfinite(cst_um) and cst_um < float(threshold_um))


def dry_sequence(cst_seq, threshold_um):
    """一條軌跡每 visit 變乾與否。cst_seq: list/array of CST（µm，可含 NaN）→ bool (T,)。"""
    return np.array([is_dry(c, threshold_um) for c in cst_seq], dtype=bool)


def recovery_event(cst_seq, threshold_um):
    """一條軌跡 → (event_step, event_observed)。

    event_step    : 第一次變乾的 visit index（0-based）；未變乾 = 最後 visit index（右設限）。
    event_observed: True=曾變乾；False=右設限（到最後沒變乾）。空軌跡 → (0, False)。
    """
    dry = dry_sequence(cst_seq, threshold_um)
    T = len(dry)
    if T == 0:
        return 0, False
    if dry.any():
        return int(np.argmax(dry)), True             # argmax 回第一個 True
    return T - 1, False                              # 設限：到最後沒變乾


def batch_recovery_events(cst_seqs, threshold_um):
    """多條軌跡 → (event_step (B,) int64, event_observed (B,) bool)。允許不等長。"""
    es, eo = [], []
    for seq in cst_seqs:
        e, o = recovery_event(seq, threshold_um)
        es.append(e); eo.append(o)
    return np.array(es, np.int64), np.array(eo, dtype=bool)


def recovery_rate(cst_seqs, threshold_um):
    """A-1 ①: 曾變乾的眼比例 + 變乾時 visit 次數分布（= event_step+1）。"""
    es, eo = batch_recovery_events(cst_seqs, threshold_um)
    n = len(eo)
    n_dry = int(eo.sum())
    visit_at_dry = (es[eo] + 1).tolist() if n_dry else []        # 第幾次 visit 變乾
    return {"n_eyes": n, "n_dry": n_dry,
            "dry_rate": (n_dry / n) if n else 0.0,
            "censored_rate": ((n - n_dry) / n) if n else 0.0,
            "visit_at_dry": visit_at_dry}


# --------------------------------------------------------------------------- #
# self-test: python forecast_c/census/recovery.py
# --------------------------------------------------------------------------- #
def _self_test():
    print("[self-test] recovery: CST 軌跡 → 變乾事件 ...")
    TH = 300.0   # 暫定乾閾值 µm（待醫師分病種定案）

    # ① 第 2 visit 變乾（350→320→280→260）→ event_step=2, observed
    assert recovery_event([350.0, 320.0, 280.0, 260.0], TH) == (2, True)
    # ② 一路濕、從沒變乾 → 右設限（最後 index, not observed）
    assert recovery_event([400.0, 380.0, 360.0], TH) == (2, False)
    # ③ NaN（層失敗）不誤判成變乾
    assert recovery_event([float("nan"), 290.0], TH) == (1, True)
    assert not is_dry(float("nan"), TH)
    # ④ batch + 普查率
    es, eo = batch_recovery_events([[350., 320., 280., 260.], [400., 380., 360.],
                                    [float("nan"), 290.]], TH)
    assert es.tolist() == [2, 2, 1] and eo.tolist() == [True, False, True]
    rate = recovery_rate([[350., 320., 280., 260.], [400., 380., 360.],
                          [float("nan"), 290.]], TH)
    assert rate["n_dry"] == 2 and abs(rate["dry_rate"] - 2 / 3) < 1e-6
    assert sorted(rate["visit_at_dry"]) == [2, 3]
    print(f"  [OK] 變乾率={rate['dry_rate']:.2f}, 變乾 visit 分布={sorted(rate['visit_at_dry'])}")
    print("[self-test OK]")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _self_test()
