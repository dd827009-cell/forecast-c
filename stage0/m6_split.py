"""
Stage 0 - Module 6: patient-level split (train/val/test = 96/2/2) + 產出 manifest。

⚠️ 切分鐵則: 同一 patient_id 的所有 visit / 雙眼必須整批進同一個 split, 否則縱向 +
雙眼資料會造成嚴重洩漏。故切分單位是「病人」, 不是「volume」。

切法 (greedy, 依 volume 數對齊比例):
  - 只在 qc_pass==True 的 volume 上切 (被 QC 丟掉的不進任何 split, 但仍寫入 manifest
    並標 split='dropped', 保留可追溯性)。
  - 目標以 volume 數計算 (train/val/test = 0.96/0.02/0.02 × 通過總數)。
  - 病人依其通過 volume 數由多到少排序, 逐一指派到「目前相對目標缺口最大」的 split,
    使小資料情境 (pilot) 也能讓 val/test 非空, 大規模時逼近 96/2/2。
  - seed 控制 tie-break, 可重現。

穩定增量 (資料陸續匯出): 用 --prev-manifest 沿用上一版分配 ——
  - 上一版已分配的病人 (train/val/test) 一律「凍結」維持原 split (其新 visit/另一眼自動跟著);
  - 只有全新病人才分配, 並以「補當下總量目標缺口」的方式維持 96/2/2 (自我修正, 非精準);
  - A2 縱向 seeding 只作用在新病人。
  目的: 同一病人永不跨 split (無洩漏) + val/test 既有成員固定 (指標跨批可比)。

輸出 stage0/manifest.parquet, 每列一顆 volume, 欄位涵蓋 pipeline 規格:
  patient_id, eye, visit_id, longitudinal_key, shard_path(M7 填), H_orig, W_orig,
  image_quality, valid_ascan_ratio, age, sex, split, transform_version, qc_flags, qc_pass

用法:
    python stage0/m6_split.py --index stage0/index_qc.parquet --out stage0/manifest.parquet
    python stage0/m6_split.py --ratios 0.96,0.02,0.02 --seed 0
    # 增量 (沿用舊分配, 只切新病人):
    python stage0/m6_split.py --index stage0/index_qc.parquet --out stage0/manifest.parquet \
        --prev-manifest stage0/manifest_prev.parquet
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from _version import STAGE0_VERSION

try:                                  # Windows 主控台預設 cp950, 統一改 utf-8 避免印中文/符號報錯
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def longitudinal_pids(passed):
    """回傳 (longi_pid_set, eye_visits)。
    longi_pid = 旗下任一眼(longitudinal_key)有 >=2 個不同 qc_pass visit 的病人;
    這種病人才提供「同眼跨 visit」的 progression pair (Stage 2 評估剛需)。"""
    eye_visits = passed.groupby("longitudinal_key")["visit_id"].nunique()
    multivisit_eyes = eye_visits[eye_visits >= 2]
    longi = set(str(k).split("::")[0] for k in multivisit_eyes.index)
    return longi, eye_visits


def load_prev_assignments(prev_path):
    """讀上一版 manifest, 回傳 dict patient_id -> split (只取已分配的 train/val/test)。
    同一病人若出現多列, 取其 (非 dropped) split (理應一致)。"""
    prev = pd.read_parquet(prev_path)
    prev = prev[prev["split"].isin(["train", "val", "test"])]
    # 每個病人取第一個出現的 split (穩定; 正常情況同病人 split 唯一)
    return prev.groupby("patient_id")["split"].first().astype(str).to_dict()


def greedy_patient_split(pat_sizes, ratios, seed=0, pre_assign=None):
    """pat_sizes: dict patient_id -> n_pass_volumes; ratios: (train,val,test)。
    pre_assign: dict pid -> split, 已預先固定 (A2 縱向 seeding), greedy 不再動它們但計入配額。
    回傳 (assign, cur, targets)。"""
    total = sum(pat_sizes.values())
    names = ["train", "val", "test"]
    targets = {n: r * total for n, r in zip(names, ratios)}
    cur = {n: 0 for n in names}
    assign = {}
    pre_assign = pre_assign or {}

    # 先把預指派 (縱向 seeds) 計入 cur, 讓後續 greedy 看得到它們已佔的配額
    for pid, sp in pre_assign.items():
        assign[pid] = sp
        cur[sp] += pat_sizes[pid]

    rng = np.random.default_rng(seed)
    pats = [(pid, sz) for pid, sz in pat_sizes.items() if pid not in pre_assign]
    rng.shuffle(pats)
    pats.sort(key=lambda kv: kv[1], reverse=True)   # 由大到小

    for pid, size in pats:
        # 指派到「缺口 (target - current) 最大」的 split; 缺口相同時偏好 target 大者
        deficit = {n: targets[n] - cur[n] for n in names}
        best = max(names, key=lambda n: (deficit[n], targets[n]))
        assign[pid] = best
        cur[best] += size
    return assign, cur, targets


def seed_longitudinal(pat_sizes, longi, per_split, seed=0):
    """A2: 優先把『最小的』縱向病人 seed 進 val/test (最小化比例失真),
    每個 eval split 最多 per_split 個。回傳 pre_assign dict 與實際 seed 數。"""
    pre = {}
    if per_split <= 0 or not longi:
        return pre, {"val": 0, "test": 0}
    rng = np.random.default_rng(seed)
    cand = sorted(((pid, pat_sizes[pid]) for pid in longi if pid in pat_sizes),
                  key=lambda kv: (kv[1], kv[0]))   # size 升冪, 同 size 用 pid 穩定排序
    counts = {"val": 0, "test": 0}
    order = ["val", "test"]
    i = 0
    for pid, _ in cand:
        # 找還沒 seed 滿的 eval split (round-robin val->test)
        placed = False
        for _ in range(2):
            sp = order[i % 2]; i += 1
            if counts[sp] < per_split:
                pre[pid] = sp; counts[sp] += 1; placed = True
                break
        if not placed and all(counts[s] >= per_split for s in order):
            break
    return pre, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="stage0/index_qc.parquet")
    ap.add_argument("--out", default="stage0/manifest.parquet")
    ap.add_argument("--ratios", default="0.96,0.02,0.02",
                    help="train,val,test (會正規化)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--longi-per-eval-split", type=int, default=1,
                    help="A2: 每個 val/test 至少 seed 幾個縱向病人(同眼>=2visit), "
                         "挑最小的以減少比例失真; 0=關閉")
    ap.add_argument("--prev-manifest", default=None,
                    help="穩定增量: 上一版 manifest。已分配病人凍結沿用, 只切新病人")
    args = ap.parse_args()

    df = pd.read_parquet(args.index).copy()
    ratios = np.array([float(x) for x in args.ratios.split(",")], dtype=float)
    ratios = ratios / ratios.sum()

    passed = df[df["qc_pass"]]
    pat_sizes = passed.groupby("patient_id").size().to_dict()
    n_pat = len(pat_sizes)
    print(f"讀入 {len(df)} 列, qc_pass {len(passed)} 顆, 病人(至少1顆通過) {n_pat}")
    if n_pat == 0:
        print("無通過 QC 的 volume, 結束。")
        return
    if n_pat < 3:
        print(f"[警告] 通過 QC 的病人僅 {n_pat} 位, 無法湊滿 train/val/test 三邊; "
              f"部分 split 將為空 (pilot 預期)。")

    # 穩定增量: 載入上一版分配, 凍結既有病人, 只對新病人切分
    frozen = {}
    if args.prev_manifest:
        if not os.path.exists(args.prev_manifest):
            print(f"[警告] --prev-manifest {args.prev_manifest} 不存在; 退化為一次性全量切分")
        else:
            prev_map = load_prev_assignments(args.prev_manifest)
            frozen = {pid: prev_map[pid] for pid in pat_sizes if pid in prev_map}
            n_new = len(pat_sizes) - len(frozen)
            print(f"\n[增量] 沿用上一版: 凍結 {len(frozen)} 位既有病人, 新病人 {n_new} 位待分配")

    # 只在「新病人」上做切分 (frozen 已固定); 全量模式時 frozen 為空, 等於全部都是新病人
    new_pat_sizes = {pid: sz for pid, sz in pat_sizes.items() if pid not in frozen}

    # A2: 縱向覆蓋。找縱向病人 (同眼>=2 visit), seed 最小的進 val/test, 確保 Stage 2
    # progression/retrieval 評估有資料。增量模式下只在「新病人」中 seed (不動凍結的)。
    longi, eye_visits = longitudinal_pids(passed)
    new_longi = {pid for pid in longi if pid in new_pat_sizes}
    seeds, seed_counts = seed_longitudinal(new_pat_sizes, new_longi,
                                           args.longi_per_eval_split, seed=args.seed)
    if args.longi_per_eval_split > 0 and new_pat_sizes:
        print(f"\n[A2] 縱向(新)病人 {len(new_longi)} 位 (同眼>=2 visit)。"
              f"seed 進 val={seed_counts['val']} / test={seed_counts['test']} "
              f"(目標各 {args.longi_per_eval_split}, 挑最小顆)")
        if seed_counts["val"] < args.longi_per_eval_split or \
           seed_counts["test"] < args.longi_per_eval_split:
            print(f"  [提醒] 新縱向病人不足以填滿 val/test seed 配額 "
                  f"(增量時既有 val/test 可能已有覆蓋)。")

    # pre_assign = 凍結病人 + 新縱向 seeds; greedy 把這些計入配額, 再分配其餘新病人
    pre_assign = {**frozen, **seeds}
    assign, cur, targets = greedy_patient_split(pat_sizes, ratios, seed=args.seed,
                                                pre_assign=pre_assign)

    # 寫回每列 split: 通過的依病人指派; 未通過的標 'dropped'
    def row_split(r):
        if not r["qc_pass"]:
            return "dropped"
        return assign.get(r["patient_id"], "dropped")
    df["split"] = df.apply(row_split, axis=1)

    # A2: 每列帶該眼的 qc_pass visit 數 (eye_n_visits)。>=2 即此 volume 屬可做 progression
    # pair 的眼; Stage 2 直接用此欄 + longitudinal_key 配對, 不必再回掃。非 qc_pass 設 NA。
    ev_map = eye_visits.to_dict()
    df["eye_n_visits"] = df.apply(
        lambda r: ev_map.get(r["longitudinal_key"]) if r["qc_pass"] else pd.NA, axis=1)

    # 組 manifest 欄位 (依 pipeline 規格命名)
    man = pd.DataFrame({
        "patient_id": df["patient_id"],
        "eye": df["laterality"],
        "visit_id": df["visit_id"],
        "longitudinal_key": df["longitudinal_key"],
        "shard_path": pd.NA,                 # 由 M7 填
        "shard_key": pd.NA,                  # 由 M7 填
        "h5_path": df["h5_path"],            # 來源追溯
        "H_orig": df["vol_h"],
        "W_orig": df["vol_w"],
        "image_quality": df["image_quality"],
        "valid_ascan_ratio": df["valid_ascan_ratio"],
        "age": df["age_at_visit_years"],
        "sex": df["sex"],
        # per-volume 物理尺度 (A1): µm 換算用。M1 已驗證全域常數, 仍逐顆帶上以防未來混入不同設備
        "scale_axial_um_per_px": df.get("scale_axial_um_per_px"),
        "scale_lateral_mm_per_px": df.get("scale_lateral_mm_per_px"),
        "scale_bscan_spacing_mm": df.get("scale_bscan_spacing_mm"),
        "eye_n_visits": df["eye_n_visits"],   # A2: 該眼 qc_pass visit 數 (>=2 可做 progression pair)
        "split": df["split"],
        "transform_version": STAGE0_VERSION,
        "qc_pass": df["qc_pass"],
        "qc_flags": df["qc_flags"],
    })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    man.to_parquet(args.out, index=False)

    # ---- 摘要 ----
    print(f"\n寫出 manifest: {args.out}  ({len(man)} 列)")
    print("\n=== Split 摘要 (volume 數) ===")
    vc = man["split"].value_counts()
    for s in ["train", "val", "test", "dropped"]:
        if s in vc:
            tgt = targets.get(s)
            extra = f"  (目標 {tgt:.1f})" if tgt is not None else ""
            print(f"  {s:8s}: {vc[s]:5d} volumes{extra}")
    print("\n=== Split 摘要 (病人數, 互斥檢查) ===")
    pat_split = man[man["qc_pass"]].groupby("split")["patient_id"].nunique()
    print(pat_split.to_string())

    # === A2: 縱向覆蓋報告 (Stage 2 progression/retrieval 可行性) ===
    print("\n=== A2 縱向覆蓋 (Stage 2 評估可行性) ===")
    mp = man[man["qc_pass"]]
    for s in ["train", "val", "test"]:
        sub = mp[mp["split"] == s]
        if sub.empty:
            print(f"  {s:6s}: (空)"); continue
        # 可做 progression pair 的眼 = 此 split 內 longitudinal_key 的 qc_pass visit 數 >=2
        ev = sub.groupby("longitudinal_key")["visit_id"].nunique()
        pair_eyes = int((ev >= 2).sum())
        # 可做 retrieval 的病人 = 此 split 內 >=2 顆 volume 的病人
        pv = sub.groupby("patient_id").size()
        retr_pats = int((pv >= 2).sum())
        print(f"  {s:6s}: {len(sub)} vols / {sub['patient_id'].nunique()} 病人 / "
              f"{len(ev)} 眼 | progression pair 眼數(同眼>=2visit)={pair_eyes} | "
              f"retrieval 病人(>=2vol)={retr_pats}")
    if mp[mp["split"] == "val"].pipe(
            lambda d: d.groupby("longitudinal_key")["visit_id"].nunique().ge(2).sum()) == 0:
        print("  [警告] val 無 progression pair 眼; Stage 2 早停的 progression 指標將無法在 val 評估。")
    if mp[mp["split"] == "test"].pipe(
            lambda d: d.groupby("longitudinal_key")["visit_id"].nunique().ge(2).sum()) == 0:
        print("  [警告] test 無 progression pair 眼; Stage 2 最終 progression 報告將受限。")

    # 洩漏自檢: 同一病人不可跨多個 (非 dropped) split
    chk = (man[man["split"] != "dropped"]
           .groupby("patient_id")["split"].nunique())
    leak = chk[chk > 1]
    if len(leak):
        print(f"\n[嚴重] {len(leak)} 位病人跨多個 split! \n", leak.to_string())
    else:
        print("\n洩漏自檢: 通過 ✓ (每位病人僅屬單一 split)")

    # 增量自檢: 凍結病人的 split 必須與上一版完全一致 (沒被改動)
    if frozen:
        cur_map = (man[man["split"] != "dropped"]
                   .groupby("patient_id")["split"].first().to_dict())
        changed = {pid: (frozen[pid], cur_map.get(pid))
                   for pid in frozen if cur_map.get(pid) != frozen[pid]}
        if changed:
            print(f"[嚴重] {len(changed)} 位凍結病人 split 被改動! 範例: "
                  f"{dict(list(changed.items())[:5])}")
        else:
            print(f"增量自檢: 通過 ✓ ({len(frozen)} 位既有病人 split 全部沿用未變)")


if __name__ == "__main__":
    main()
