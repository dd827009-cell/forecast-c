"""Phase 1b 跨切片 3D adapter — vendored from 吳韋論。

來源: `Desktop/吳韋論/program-wl/models/utils/peft.py:Adapter3D`（吳韋論碩論「雙分支+PEFT」
      消融最強元件）。本檔 = 參數化 + 殘差零初始（起點恆等）+ 注入凍結 2D 學生的 helper。

機制（對上設計 C §4「跨切片 = 3D adapter 插進凍結 2D 學生」）:
  - **depth-only `Conv3d(kernel=(3,1,1))`**: 只沿切片(深度)軸混、in-slice 1×1 → 保住切片內空間特徵
    不被擾動。**別用 3×3×3**（會動到 in-slice）。
  - bottleneck（linear down → 3D conv → GELU → linear up）+ residual + LayerNorm。
  - 學生仍是 2D per-slice：一個 3D volume 的 d_size 張 B-scan 攤平進 batch（B_vol×d_size, N, C），
    adapter 在 d_size 軸做 3×1×1 conv 把跨切片資訊混回來。base 凍結、只訓 adapter（PEFT）。
  - 殘差零初始（up 投影零初始）→ Δ=0 起點恆等 → adapter on/off = 天然消融。

與吳韋論差異: 他用在「分類微調」；設計 C 用在 **空間 JEPA SSL 訓 adapter**（見 spatial_jepa.py）。
"""
import torch
import torch.nn as nn


class Adapter3D(nn.Module):
    """跨切片 3D adapter（depth-only 3×1×1）。forward(x (B, N[, +1 cls], C)) -> (同形狀)。

    dim             : token 維度（= 學生 embed_dim）。
    adapter_channels: bottleneck 寬度（吳韋論 proposed=64）。
    d_size          : 一個 volume 的切片數（B 維 = n_volumes × d_size，吳韋論=16）。
    kernel_t        : 深度卷積核（3 = 3×1×1）。
    has_cls         : tokens 是否含前置 CLS（含則 CLS 單獨過同一 conv）。
    zero_init       : up 投影零初始 → 起點恆等（設計 C 要；吳韋論原版非零初始）。
    """

    def __init__(self, dim, adapter_channels=64, d_size=16, kernel_t=3,
                 has_cls=True, zero_init=True):
        super().__init__()
        self.dim, self.adapter_channels, self.d_size = dim, adapter_channels, d_size
        self.has_cls = has_cls
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, adapter_channels, bias=False)
        self.conv = nn.Conv3d(adapter_channels, adapter_channels,
                              kernel_size=(kernel_t, 1, 1), padding="same")
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_channels, dim, bias=False)
        if zero_init:
            nn.init.zeros_(self.up.weight)        # 起點恆等：adapter 輸出 0 → x+0=x

    def _conv_grid(self, x, hw):
        """x: (B, hw*hw, C_a) → 3D conv over (d_size, hw, hw) → 同形狀。"""
        B, L, Ca = x.shape
        nv = B // self.d_size
        x = x.view(nv, self.d_size, hw, hw, Ca).permute(0, 4, 1, 2, 3)   # (nv,Ca,d,hw,hw)
        x = self.conv(x)
        return x.permute(0, 2, 3, 4, 1).reshape(B, hw * hw, Ca)

    def _conv_cls(self, x_cls):
        """CLS: (B,1,C_a) → 當 (d_size,1,1) 過同一 conv。"""
        B = x_cls.shape[0]
        nv, Ca = B // self.d_size, self.adapter_channels
        x = x_cls.view(nv, self.d_size, 1, 1, Ca).permute(0, 4, 1, 2, 3)
        x = self.conv(x)
        return x.permute(0, 2, 3, 4, 1).reshape(B, 1, Ca)

    def forward(self, x):
        assert x.shape[0] % self.d_size == 0, \
            f"batch({x.shape[0]}) 必須是 d_size({self.d_size}) 的倍數（= n_volumes×d_size）"
        shortcut = x
        h = self.down(self.norm(x))
        if self.has_cls:
            cls, patches = h[:, :1], h[:, 1:]
            hw = int(round(patches.shape[1] ** 0.5))
            assert hw * hw == patches.shape[1], "patch token 數須為完全平方（per-slice 2D 網格）"
            patches = self._conv_grid(patches, hw)
            cls = self._conv_cls(cls)
            h = torch.cat([cls, patches], dim=1)
        else:
            hw = int(round(h.shape[1] ** 0.5))
            assert hw * hw == h.shape[1], "patch token 數須為完全平方"
            h = self._conv_grid(h, hw)
        return shortcut + self.up(self.act(h))            # residual


class AdaptedBlock(nn.Module):
    """把凍結的 2D 學生 block 包成「block(x) → Adapter3D」。block 凍結、只訓 adapter。"""

    def __init__(self, block: nn.Module, adapter: Adapter3D, freeze_block=True):
        super().__init__()
        self.block = block
        self.adapter = adapter
        if freeze_block:
            for p in self.block.parameters():
                p.requires_grad_(False)

    def forward(self, x, *args, **kwargs):
        return self.adapter(self.block(x, *args, **kwargs))


def inject_adapter3d(blocks: nn.ModuleList, dim, d_size, adapter_channels=64,
                     has_cls=True, kernel_t=3):
    """把每個學生 block 換成 AdaptedBlock（base 凍、adapter 可訓）。回傳 adapter 參數 list。

    blocks: 凍結 2D 學生的 transformer blocks（nn.ModuleList，逐 block 介面 block(x)->x）。
    adapter on/off = 天然消融（不注入 = 純 2D；注入 = 跨切片感知）。
    """
    adapter_params = []
    for i, blk in enumerate(blocks):
        ad = Adapter3D(dim, adapter_channels, d_size, kernel_t, has_cls=has_cls)
        blocks[i] = AdaptedBlock(blk, ad)
        adapter_params += list(ad.parameters())
    return adapter_params


# ───────────────────────── dummy 自測（`python -m forecast_c.phase1.adapter3d`） ─────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    dim, d_size, hw = 64, 4, 4               # 每 slice 4×4=16 patch + 1 cls
    nv, N = 2, hw * hw + 1
    x = torch.randn(nv * d_size, N, dim)     # (B=nv*d_size, 1+16, dim)

    ad = Adapter3D(dim, adapter_channels=16, d_size=d_size, has_cls=True)
    y = ad(x)
    assert y.shape == x.shape
    # 殘差零初始 → 起點恆等（adapter 不改變輸入）
    assert torch.allclose(y, x, atol=1e-6), "zero_init 應起點恆等"
    # 訓練後（給 up 非零）→ 跨切片真的混（改某 volume 的一張切片，會影響鄰切片輸出）
    nn.init.normal_(ad.up.weight, std=0.1)
    y2 = ad(x)
    assert not torch.allclose(y2, x, atol=1e-4), "adapter 應改變輸出"
    # 非均勻擾動（LayerNorm 不會像均勻位移那樣抹掉）改 volume0 的 slice0
    x_perturb = x.clone(); x_perturb[0] = x[0] + torch.randn_like(x[0]) * 2.0
    y3 = ad(x_perturb)
    assert not torch.allclose(y3[1], y2[1], atol=1e-4), "depth conv 應讓鄰切片(slice1)受 slice0 影響"
    assert torch.allclose(y3[d_size], y2[d_size], atol=1e-6), "不同 volume 不該互相影響"
    y2.sum().backward(); assert ad.up.weight.grad is not None

    # inject helper: 凍結 dummy 學生 + 注入 adapter
    blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
    for p in (p for b in blocks for p in b.parameters()):
        p.requires_grad_(False)
    ap = inject_adapter3d(blocks, dim, d_size, adapter_channels=16, has_cls=True)
    assert len(ap) > 0 and all(p.requires_grad for p in ap)
    out = x
    for b in blocks:
        out = b(out)
    assert out.shape == x.shape
    print("adapter3d dummy 自測通過 ✅（depth-only 3×1×1、起點恆等、跨切片混、volume 不串）")
