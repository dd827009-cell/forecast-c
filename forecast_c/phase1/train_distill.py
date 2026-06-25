"""Phase 1a 訓練迴圈 — 多教師蒸餾 + MIM（楊瀚博 FoundMIM，OCT 域）。

流程（治療無關）:
  B-scan → 教師特徵（DINOv3 + RETFound，凍結，on-the-fly）→ FoundMIM 學生 forward
    （遮 50% → 編可見 → cross-attn decoder 補 → translator 轉教師格式）→ cosine loss + MAE 重建。
  只訓學生（教師凍結）。訓完丟 decoder/translator，留 ViT 學生 → Phase 1b。

⚠️ MedSAM2 教師待接 sam2（先用 2 教師）。L40 應**預計算**教師特徵存檔（楊瀚博式，省重算）；
   本檔 on-the-fly 適合 smoke / 小規模。

smoke（pilot，CPU）: python -m forecast_c.phase1.train_distill --smoke --steps 2
"""
import argparse
import glob

import torch
from torch.utils.data import Dataset, DataLoader

from forecast_c.data.oct_h5 import read_h5
from .distill import build_foundmim_for_oct, OCT_TEACHERS
from .teachers import TeacherFeatureExtractor

DISTILL_TEACHERS = ("dinov3_vitl16", "retfound_oct", "medsam2")    # 三教師（MedSAM2 需 sam2）


class BscanDataset(Dataset):
    """h5_output 的每張 B-scan 當一個蒸餾樣本。回傳 raw B-scan (1, H, W)。"""

    def __init__(self, h5_dir, max_studies=None, max_per_study=None):
        self.items = []                                  # (file, bscan_idx)
        files = sorted(glob.glob(f"{h5_dir}/**/*.h5", recursive=True))
        if max_studies:
            files = files[:max_studies]
        for f in files:
            n = read_h5(f)["volume"].shape[0]
            idxs = range(n if max_per_study is None else min(n, max_per_study))
            self.items += [(f, i) for i in idxs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, bi = self.items[i]
        vol = read_h5(f)["volume"]                       # (25,496,512)
        return torch.from_numpy(vol[bi]).float().unsqueeze(0)   # (1,496,512)


def train_step(student, ext, raw, mask_ratio, teachers):
    """raw (B,1,H,W) → 教師特徵 + 學生 forward → total loss + detail。"""
    imgs = ext._preprocess(raw)                          # (B,3,224,224) 與教師同前處理
    teacher_features = {t: ext.extract(t, raw) for t in teachers}    # {t:[B,196,1024]}（stop-grad）
    loss_mae, loss_teacher, loss_bal, _, _ = student(imgs, mask_ratio=mask_ratio,
                                                     teacher_features=teacher_features)
    total = loss_mae if loss_mae is not None else 0.0
    detail = {"mae": float(loss_mae) if loss_mae is not None else 0.0}
    for t, d in loss_teacher.items():
        total = total + d["patch"]
        detail[t] = float(d["patch"])
    return total, detail


def main():
    ap = argparse.ArgumentParser(description="Phase 1a 多教師蒸餾訓練（FoundMIM）")
    ap.add_argument("--smoke", action="store_true", help="pilot 小規模驗迴圈")
    ap.add_argument("--h5-dir", default="h5_output")
    ap.add_argument("--ckpt-dir", default="ckpts")
    ap.add_argument("--student", default="small", choices=["small", "base", "large"])
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--save", help="訓完存 2D 學生 ViT(餵 Phase 1b)；smoke 預設存 ckpts/phase1a_student_smoke.pth")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teachers = list(DISTILL_TEACHERS)
    student, mr = build_foundmim_for_oct(
        student=a.student, teachers=teachers, img_size=224, mask_ratio=a.mask_ratio,
        teacher_feature_sizes={t: OCT_TEACHERS[t] for t in teachers})
    student.to(device).train()
    ext = TeacherFeatureExtractor(ckpt_dir=a.ckpt_dir, img_size=224, device=device)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr)

    ms = 2 if a.smoke else None
    ds = BscanDataset(a.h5_dir, max_studies=ms, max_per_study=(4 if a.smoke else None))
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True)
    print(f"[distill] device={device} 學生={a.student} 教師={teachers} 樣本={len(ds)} B-scan")

    step = 0
    for raw in loader:
        raw = raw.to(device)
        total, detail = train_step(student, ext, raw, mr, teachers)
        opt.zero_grad(); total.backward(); opt.step()
        print(f"  step {step}: total={float(total):.4f} {detail}")
        step += 1
        if step >= a.steps:
            break

    # 存 2D 學生 ViT（patch_embed/blocks/pos_embed/cls_token/norm）→ 餵 Phase 1b
    save = a.save or ("ckpts/phase1a_student_smoke.pth" if a.smoke else None)
    if save:
        enc = {k: v for k, v in student.state_dict().items()
               if k.startswith(("patch_embed", "blocks", "norm", "cls_token", "pos_embed"))}
        torch.save({"student": enc, "arch": a.student, "img_size": 224}, save)
        print(f"[distill] 存 2D 學生 → {save}（{len(enc)} 鍵）")
    print("[distill] Phase 1a 訓練迴圈跑通 ✅（教師凍結、只訓學生）")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
