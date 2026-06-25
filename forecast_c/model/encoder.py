"""凍結 encoder — 最小版 = 官方 OCTCube（現成 3D，不蒸餾）。

契約（整個 forecast_c 共用）:
    encoder(volume) -> (tokens (B, N, D), cls (B, D))
    tokens = 256 語意分支 token 網格 = z_t / 預測目標所在；cls = 全局 token。

最小版 encoder **完全凍結**（DINO-WM 式：只學動力學，不動 backbone）。target 也用
**同一個凍結 encoder** 編未來（detach + LayerNorm，stop-grad 自動成立）→ ★ 不需要 EMA teacher。

三塊:
  - FrozenEncoder         : 把任一 inner encoder 凍結 + 正規化輸出成 (tokens, cls)。
  - build_octcube_encoder : 接官方 OCTCube（models_vit_st_joint）+ 載 ckpts/OCTCube.pth（flash→非flash remap）。
  - DummyEncoder          : 線性假 encoder，供本機 dummy 單元測（無權重/無 GPU）。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_tokens_cls(out, cls_embed=True):
    """把 encoder 原始輸出正規化成 (tokens, cls)。

    支援 inner 回傳:
      - (tokens, cls)                 → 直接用
      - tokens (B, 1+N, D)            → 切出 cls=tokens[:,0]、其餘為網格（cls_embed=True）
      - tokens (B, N, D)              → 無 cls（cls=None）
    """
    if isinstance(out, (tuple, list)):
        return out[0], (out[1] if len(out) > 1 else None)
    if cls_embed and out.dim() == 3:
        return out[:, 1:], out[:, 0]
    return out, None


class FrozenEncoder(nn.Module):
    """凍結任一 inner encoder + 正規化輸出成 (tokens, cls)。

    inner: callable(volume) -> (tokens, cls) | tokens。建構即凍結（requires_grad=False + eval）。
    """

    def __init__(self, inner: nn.Module, cls_embed: bool = True):
        super().__init__()
        self.inner = inner
        self.cls_embed = cls_embed
        for p in self.inner.parameters():
            p.requires_grad_(False)
        self.inner.eval()

    def train(self, mode: bool = True):
        # 永遠保持 eval（凍結 encoder 不該因 model.train() 切回 train，免動 BN/dropout 統計）
        super().train(mode)
        self.inner.eval()
        return self

    @torch.no_grad()
    def forward(self, volume):
        return _split_tokens_cls(self.inner(volume), self.cls_embed)


class DummyEncoder(nn.Module):
    """本機 dummy 單元測用假 encoder: volume (B, N, D_in) → (tokens (B,N,D), cls (B,D))。

    不載權重、不需 GPU/flash-attn。線性投影 + 平均當 cls。供 FrozenEncoder 包起來測流程。
    """

    def __init__(self, embed_dim: int, in_dim: int = None):
        super().__init__()
        in_dim = in_dim or embed_dim
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, volume):
        tokens = self.proj(volume)
        return tokens, tokens.mean(dim=1)


def _octcube_remap(sd):
    """OCTCube.pth(flash 命名) → 非 flash st_joint 鍵。

    flash MHA 的 `blocks.*.mixer.Wqkv`(fused 3072×1024) → 非 flash 的 `attn.q/k/v`(各 1024×1024);
    `mixer.out_proj` → `attn.proj`。丟 decoder_* / mask_token(encoder 不需)。實測 0 unexpected。
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("decoder") or k == "mask_token":
            continue
        if ".mixer.Wqkv." in k:
            kind = "weight" if k.endswith("weight") else "bias"
            q, kk, vv = v.chunk(3, dim=0)
            base = k.replace(".mixer.Wqkv." + kind, "")
            out[base + ".attn.q." + kind] = q
            out[base + ".attn.k." + kind] = kk
            out[base + ".attn.v." + kind] = vv
        elif ".mixer.out_proj." in k:
            out[k.replace(".mixer.out_proj.", ".attn.proj.")] = v
        else:
            out[k] = v
    return out


class OCTCubeTokenEncoder(nn.Module):
    """包官方 OCTCube st_joint VisionTransformer → 回 (tokens, cls)（token 網格，非 pool/head）。

    輸入 volume: (B, 1, num_frames, img, img)。256×256×60frames → 20×16×16 = 5120 token, D=1024。
    forward 複製官方 forward 到 blocks+norm 為止（跳過 global_pool / head）。
    """

    def __init__(self, model, num_frames=60, img_size=256):
        super().__init__()
        self.m = model
        self.num_frames, self.img_size = num_frames, img_size

    @staticmethod
    def prep(volume, num_frames=60, img_size=256):
        """OCT cube → 模型輸入。volume: (T,H,W) np/tensor → (1,1,num_frames,img,img)，min-max[0,1]。"""
        x = torch.as_tensor(volume, dtype=torch.float32)
        if x.dim() == 3:
            x = x[None, None]                                  # (1,1,T,H,W)
        x = F.interpolate(x, size=(num_frames, img_size, img_size),
                          mode="trilinear", align_corners=False)
        lo, hi = x.amin(), x.amax()
        return (x - lo) / (hi - lo + 1e-6)

    def forward(self, x):
        """x: (B,1,num_frames,img,img) → (tokens (B,5120,1024), cls (B,1024))。"""
        m = self.m
        H = x.shape[-2]
        high_res = (H == m.high_res_input_size[1] * m.high_res_patch_embed.patch_size[0])
        x = m.high_res_patch_embed(x) if high_res else m.patch_embed(x)
        N, T, L, C = x.shape
        x = x.view([N, T * L, C])
        if m.cls_embed:
            x = torch.cat((m.cls_token.expand(N, -1, -1), x), dim=1)
        # sep pos embed（低解析時把 spatial pos 內插到 input size）
        if not high_res:
            pos = F.interpolate(
                m.pos_embed_spatial.view(1, m.high_res_input_size[1], m.high_res_input_size[2], -1)
                .permute(0, 3, 1, 2), [m.input_size[1], m.input_size[2]],
                mode="bicubic", align_corners=False).permute(0, 2, 3, 1).view(1, m.input_size[1] * m.input_size[2], -1)
            pos_h, pos_w = m.input_size[1], m.input_size[2]
        else:
            pos = m.pos_embed_spatial
            pos_h, pos_w = m.high_res_input_size[1], m.high_res_input_size[2]
        pos = pos.repeat(1, T, 1) + torch.repeat_interleave(m.pos_embed_temporal, pos_h * pos_w, dim=1)
        if m.cls_embed:
            pos = torch.cat([m.pos_embed_class.expand(pos.shape[0], -1, -1), pos], 1)
        x = x + pos
        for blk in m.blocks:
            x = blk(x)
        x = m.norm(x)                                          # token 網格 latent（含 cls）
        return x[:, 1:], x[:, 0]                               # (tokens, cls)


def build_octcube_encoder(ckpt_path="ckpts/OCTCube.pth", repo_dir="OCTCubeM-main/OCTCube",
                          img_size=256, num_frames=60, device="cpu", freeze=True):
    """接官方 OCTCube → FrozenEncoder。載 ckpts/OCTCube.pth（flash→非flash remap），凍結。

    256×256×60frames → 5120 token×1024（對齊 BackboneSpec）。CPU 可跑（慢）；L40 裝 flash-attn 更快。
    回傳 FrozenEncoder（forward(volume)->(tokens,cls)）。
    """
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import models_vit_st_joint as M                            # 官方非 flash st_joint
    model = M.vit_large_patch16(
        num_frames=num_frames, t_patch_size=3, img_size=img_size, in_chans=1,
        sep_pos_embed=True, cls_embed=True, global_pool=True, num_classes=0,
        use_high_res_patch_embed=True, high_res_img_size=512)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    msg = model.load_state_dict(_octcube_remap(sd), strict=False)
    miss = [k for k in msg.missing_keys if not k.startswith(("head", "decoder"))]
    assert not miss, f"OCTCube 載入缺實質鍵: {miss[:8]}"
    model.eval().to(device)
    inner = OCTCubeTokenEncoder(model, num_frames, img_size)
    return FrozenEncoder(inner) if freeze else inner


# ───────────────────────── dummy 自測（容器內 `python -m forecast_c.model.encoder`） ─────────────────────────
if __name__ == "__main__":
    from forecast_c.config import ForecastConfig
    cfg = ForecastConfig.tiny()
    N, D = cfg.backbone.n_tokens, cfg.backbone.embed_dim

    enc = FrozenEncoder(DummyEncoder(D))
    # 凍結確認
    assert all(not p.requires_grad for p in enc.parameters())
    tokens, cls = enc(torch.randn(4, N, D))
    assert tokens.shape == (4, N, D) and cls.shape == (4, D)
    assert tokens.grad_fn is None, "凍結 encoder 輸出應 stop-grad"
    # model.train() 不應解凍 inner
    enc.train()
    assert not enc.inner.training, "凍結 encoder 應恆 eval"

    # _split_tokens_cls 三種輸入
    t, c = _split_tokens_cls((torch.randn(2, N, D), torch.randn(2, D)))
    assert t.shape == (2, N, D) and c.shape == (2, D)
    t, c = _split_tokens_cls(torch.randn(2, 1 + N, D), cls_embed=True)
    assert t.shape == (2, N, D) and c.shape == (2, D)
    t, c = _split_tokens_cls(torch.randn(2, N, D), cls_embed=False)
    assert t.shape == (2, N, D) and c is None
    print("encoder dummy 自測通過 ✅")
