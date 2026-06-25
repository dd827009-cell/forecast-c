"""
Cross-attention block implementation based on timm.models.vision_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from timm.layers import Mlp, DropPath, use_fused_attn
from timm.models.vision_transformer import LayerScale


class CrossAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        encoder_dim,
        decoder_dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert decoder_dim % num_heads == 0, "decoder_dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = decoder_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(decoder_dim, decoder_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, fix_encoding_tokens, ids_restore):
        # merge encoding tokens and x
        x_ = torch.cat([fix_encoding_tokens[:, 1:, :], x], dim=1)
        assert x_.shape[1] == ids_restore.shape[1]
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1]))  # unshuffle
        x_ = torch.cat([fix_encoding_tokens[:, :1, :], x_], dim=1)

        # full sequence shape
        B, N, C = x_.shape

        # qkv
        q = self.q(x).reshape(B, x.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x_).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, x.shape[1], C)

        # output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        encoder_dim,
        decoder_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(decoder_dim)
        self.attn = CrossAttention(
            encoder_dim,
            decoder_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = LayerScale(decoder_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(decoder_dim)
        self.mlp = mlp_layer(
            in_features=decoder_dim,
            hidden_features=int(decoder_dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(decoder_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, fix_encoding_tokens, ids_restore):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), fix_encoding_tokens, ids_restore)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
