"""Phase 1b 訓練迴圈 — 空間 JEPA 訓跨切片 Adapter3D（★設計 C 原創，治療無關）。

流程:
  凍結 2D 學生（Phase 1a 產出）→ 每 volume 的 25 張 B-scan 各自 patch_embed → per-slice tokens
  → SpatialJEPA（遮 k 切片 → 帶 Adapter3D 學生從鄰切片預測被遮切片的凍結特徵，stop-grad）
  → 只訓 adapter + mask_token（base 凍）。adapter on/off = 天然消融。

smoke（pilot，CPU）: python -m forecast_c.phase1.train_jepa --smoke --steps 2
注: smoke 用新建（未訓）2D 學生只驗迴圈機制；正式應載 Phase 1a 訓好的學生。
"""
import argparse
import glob
import os

import torch

from forecast_c.data.oct_h5 import read_h5
from .teachers import TeacherFeatureExtractor
from .distill import build_foundmim_for_oct
from .spatial_jepa import SpatialJEPA


def embed_slices(student, bscans):
    """FoundMIM 2D 學生的 patch_embed + pos_embed + cls → per-slice tokens (B, 1+196, C)。"""
    x = student.patch_embed(bscans)                          # (B,196,C)
    x = x + student.pos_embed[:, 1:, :]
    cls = (student.cls_token + student.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
    return torch.cat([cls, x], dim=1)                        # (B,197,C)


def main():
    ap = argparse.ArgumentParser(description="Phase 1b 空間 JEPA 訓 Adapter3D")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--h5-dir", default="h5_output")
    ap.add_argument("--ckpt-dir", default="ckpts")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--student-ckpt", help="Phase 1a 存的 2D 學生（不給則用未訓學生只驗機制）；"
                                           "smoke 自動找 ckpts/phase1a_student_smoke.pth")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2D 學生（正式: 載 Phase 1a checkpoint；無則新建只驗機制）。teachers=() → 純 ViT。
    ckpt = a.student_ckpt or ("ckpts/phase1a_student_smoke.pth" if a.smoke else None)
    arch = "small"
    if ckpt and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        arch = blob.get("arch", "small")
    student, _ = build_foundmim_for_oct(student=arch, teachers=(), img_size=224,
                                        reconstruct_orig_img=True)
    if ckpt and os.path.exists(ckpt):
        msg = student.load_state_dict(blob["student"], strict=False)
        loaded = len(blob["student"]) - len([k for k in blob["student"] if k in msg.unexpected_keys])
        print(f"[jepa] 載入 Phase 1a 學生 {ckpt}（{loaded} 鍵）")
    else:
        print("[jepa] ⚠️ 無 Phase 1a checkpoint → 用未訓學生只驗迴圈機制")
    student.to(device).eval()
    for p in student.parameters():
        p.requires_grad_(False)                              # 2D 學生凍結（Phase 1b 只訓 adapter）
    dim = student.pos_embed.shape[-1]

    pre = TeacherFeatureExtractor(ckpt_dir=a.ckpt_dir, img_size=224, device=device)._preprocess
    files = sorted(glob.glob(f"{a.h5_dir}/**/*.h5", recursive=True))
    if a.smoke:
        files = files[:2]

    jepa = SpatialJEPA(student.blocks, dim, d_size=25, has_cls=True,
                       mask_frac=a.mask_frac).to(device)
    opt = torch.optim.AdamW(jepa.trainable_params(), lr=a.lr)
    print(f"[jepa] device={device} dim={dim} d_size=25 可訓張量={len(jepa.trainable_params())}（只 adapter+mask）")

    step = 0
    for f in files:
        vol = read_h5(f)["volume"]                           # (25,496,512)
        bscans = pre(torch.from_numpy(vol).float().unsqueeze(1).to(device))   # (25,3,224,224)
        with torch.no_grad():
            tokens = embed_slices(student, bscans)           # (25,197,C) 凍結 embed
        out = jepa(tokens)                                   # 遮切片→預測（只 adapter 有梯度）
        opt.zero_grad(); out["loss"].backward(); opt.step()
        print(f"  step {step}: loss={float(out['loss']):.4f} n_masked={out['n_masked']}/25")
        step += 1
        if step >= a.steps:
            break
    # 確認只有 adapter 在動
    assert all(p.grad is None for p in student.parameters()), "2D 學生不該有梯度"
    print("[jepa] Phase 1b 空間 JEPA 迴圈跑通 ✅（base 凍、只訓 Adapter3D；★設計C原創）")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
