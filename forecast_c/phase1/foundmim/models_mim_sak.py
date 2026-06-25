# You can try this secret implementation (this was mentioned in week meeting, but not in the thesis).
# This is a experimental implementation with SAK (Swiss Army Knife) architecture.
# The original thesis implementation is in "/program/pretrain/mim_modeling/models_mim.py".
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MILAN: https://github.com/zejiangh/MILAN
# DMAE: https://github.com/UCSC-VLAA/DMAE
# EfficientSAM: https://github.com/yformer/EfficientSAM
# AM-RADIO: https://github.com/NVlabs/RADIO
# Theia: https://github.com/bdaiinstitute/theia
# SAK: https://github.com/innovator-zero/SAK
# --------------------------------------------------------

from functools import partial

import math
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from timm.models.vision_transformer import PatchEmbed, Block
from torch.nn.functional import interpolate

from .pos_embed import get_2d_sincos_pos_embed
from .cross_attn import CrossAttentionBlock


class Adapter(nn.Module):

    def __init__(self, input_dim: int, down_ratio: int, output_dim: int = None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        hidden_dim = int(input_dim // down_ratio)
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, output_dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.scale * self.up(self.act(self.down(self.norm(x))))


class Interpolation(nn.Module):
    """Interpolation nn.Module wrap for nn.functional.interpolate.

    Attributes:
        target_size (tuple[int, int] | torch.Size): target spatial size of this interpolation.
    """

    def __init__(self, target_size: tuple[int, int] | torch.Size) -> None:
        super().__init__()
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Very simple forward pass to call interpolate()."""
        return interpolate(x, self.target_size, mode="bilinear", align_corners=False)


class FeaturePredictionHead(nn.Module):
    """
    A simple interpolation followed by a linear layer to project the feature to the target dimension.
    """

    def __init__(self, source_size, target_size):
        super().__init__()
        source_dim, source_h, source_w = source_size
        target_dim, target_h, target_w = target_size

        layers = []
        if (source_h, source_w) != (target_h, target_w):
            layers.extend(
                [
                    Rearrange("b (h w) c -> b c h w", h=source_h, w=source_w),
                    Interpolation((target_h, target_w)),
                    Rearrange("b c h w -> b (h w) c"),
                ]
            )

        layers.append(nn.Linear(source_dim, target_dim, bias=True))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class MaskedImageModelingViT(nn.Module):
    """Masked Image Modeling with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        # --------------------------------------------------------------------------
        # image reconstruction specifics
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        cross_attn_decoder=False,  # use cross-attention in decoder for image reconstruction
        reconstruct_orig_img=True,  # reconstruct the original image or not
        # --------------------------------------------------------------------------
        # teacher distillation specific
        teacher_names=None,  # e.g., ["dinov2_vitl14", "dfn_clip_vitl14"]
        loss_weights_teacher=None,  # e.g., [1.0, 1.0] for two teachers
        loss_type_teacher="cos_sim",  # loss type for teacher distillation
        direct_distillation=False,  # use direct distillation (no decoder)
        tsap_down_ratio=4,  # down-sampling ratio for teacher specific adapter path
        decoder_embed_dim_teacher=512,
        decoder_depth_teacher=8,
        decoder_num_heads_teacher=16,
        norm_pix_loss_teacher=False,
        cross_attn_decoder_teacher=False,  # use cross-attention in decoder for teacher feature reconstruction
        feature_sizes_teacher=None,  # e.g., [[1024, 16, 16], [1024, 16, 16]] for two ViT-Large teachers
        # --------------------------------------------------------------------------
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # Encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics (for original image reconstruction)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.reconstruct_orig_img = reconstruct_orig_img
        if self.reconstruct_orig_img:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.cross_attn_decoder = cross_attn_decoder
            if self.cross_attn_decoder:
                self.decoder_blocks = nn.ModuleList(
                    [
                        CrossAttentionBlock(
                            decoder_embed_dim,
                            decoder_embed_dim,
                            decoder_num_heads,
                            mlp_ratio,
                            qkv_bias=True,
                            norm_layer=norm_layer,
                        )
                        for i in range(decoder_depth)
                    ]
                )
            else:
                self.decoder_blocks = nn.ModuleList(
                    [
                        Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                        for i in range(decoder_depth)
                    ]
                )

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)  # decoder to patch

            self.norm_pix_loss = norm_pix_loss
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # VFMs distillation specifics
        self.num_teachers = 0
        if teacher_names is not None:
            self.num_teachers = len(teacher_names)
            self.teacher_names = teacher_names

            assert loss_weights_teacher is not None
            assert feature_sizes_teacher is not None

            assert len(loss_weights_teacher) == len(
                teacher_names
            ), "Please provide distillation loss weight for each teacher"
            assert len(feature_sizes_teacher) == len(
                teacher_names
            ), "Please provide projection head dim for each teacher"

            self.loss_type_teacher = loss_type_teacher
            self.loss_weights_teacher = {t: w for t, w in zip(teacher_names, loss_weights_teacher)}

            # Teacher-Specific Adapter Path
            # Borrowed from "Swiss Army Knife: Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learning"
            # Reference: https://arxiv.org/abs/2410.14633
            # https://github.com/innovator-zero/SAK/blob/main/models/backbones/sak.py#L187
            self.tsap = nn.ModuleDict()
            self.tsap_norm = nn.ModuleDict()
            for teacher_name in self.teacher_names:
                self.tsap[teacher_name] = nn.ModuleList()
                self.tsap[teacher_name].append(Adapter(embed_dim, tsap_down_ratio))  # for patch embed
                for i in range(len(self.blocks)):  # for transformer blocks
                    self.tsap[teacher_name].append(Adapter(embed_dim, tsap_down_ratio))
                self.tsap_norm[teacher_name] = norm_layer(embed_dim)

            self.direct_distillation = direct_distillation
            if not self.direct_distillation:
                # decoder for teacher feature reconstruction
                self.cross_attn_decoder_teacher = cross_attn_decoder_teacher

                self.decoder_pos_embed_teacher = nn.Parameter(
                    torch.zeros(1, num_patches + 1, decoder_embed_dim_teacher), requires_grad=False
                )  # fixed sin-cos embedding (shared between teachers)

                # Teacher Specific Decoder
                self.decoder_embed_teacher = nn.ModuleDict()
                self.mask_token_teacher = nn.ParameterDict()
                self.decoder_blocks_teacher = nn.ModuleDict()
                self.decoder_norm_teacher = nn.ModuleDict()
                for teacher_name in self.teacher_names:
                    self.decoder_embed_teacher[teacher_name] = nn.Linear(
                        embed_dim, decoder_embed_dim_teacher, bias=True
                    )
                    self.mask_token_teacher[teacher_name] = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim_teacher))

                    if self.cross_attn_decoder_teacher:
                        self.decoder_blocks_teacher[teacher_name] = nn.ModuleList(
                            [
                                CrossAttentionBlock(
                                    decoder_embed_dim_teacher,
                                    decoder_embed_dim_teacher,
                                    decoder_num_heads_teacher,
                                    mlp_ratio,
                                    qkv_bias=True,
                                    norm_layer=norm_layer,
                                )
                                for i in range(decoder_depth_teacher)
                            ]
                        )
                    else:
                        self.decoder_blocks_teacher[teacher_name] = nn.ModuleList(
                            [
                                Block(
                                    decoder_embed_dim_teacher,
                                    decoder_num_heads_teacher,
                                    mlp_ratio,
                                    qkv_bias=True,
                                    norm_layer=norm_layer,
                                )
                                for i in range(decoder_depth_teacher)
                            ]
                        )

                    self.decoder_norm_teacher[teacher_name] = norm_layer(decoder_embed_dim_teacher)

            self.pred_head_teacher = nn.ModuleDict()
            source_size = (
                embed_dim if self.direct_distillation else decoder_embed_dim_teacher,
                int(math.sqrt(num_patches)),
                int(math.sqrt(num_patches)),
            )
            for teacher_name, target_size in zip(teacher_names, feature_sizes_teacher):
                self.pred_head_teacher[teacher_name] = FeaturePredictionHead(source_size, target_size)

            self.norm_pix_loss_teacher = norm_pix_loss_teacher

        else:
            assert reconstruct_orig_img, "Either provide teacher_names or set reconstruct_orig_img to True"
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        if self.reconstruct_orig_img:
            decoder_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
            )
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        if self.num_teachers > 0 and not self.direct_distillation:
            decoder_pos_embed_teacher = get_2d_sincos_pos_embed(
                self.decoder_pos_embed_teacher.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
            )
            self.decoder_pos_embed_teacher.data.copy_(torch.from_numpy(decoder_pos_embed_teacher).float().unsqueeze(0))
            for teacher_name in self.teacher_names:
                torch.nn.init.normal_(self.mask_token_teacher[teacher_name], std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep, ids_mask

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio > 0:
            # masking: length -> length * mask_ratio
            x, mask, ids_restore, ids_keep, ids_mask = self.random_masking(x, mask_ratio)
        else:  # pure distillation
            mask = ids_restore = ids_keep = ids_mask = None

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Teacher-Specific Adatper Path
        tsap_res_dict = {}  # Dict of residuals for each adapter
        for teacher_name in self.teacher_names:
            tsap_res_dict[teacher_name] = self.tsap[teacher_name][0](x)  # patch embedding adapter

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            # Get adapter features for each teacher
            for teacher_name in self.teacher_names:
                tsap_res_dict[teacher_name] = self.tsap[teacher_name][i + 1](tsap_res_dict[teacher_name] + x)

        x = self.norm(x)
        for teacher_name in self.teacher_names:
            tsap_res_dict[teacher_name] = self.tsap_norm[teacher_name](tsap_res_dict[teacher_name])

        return x, tsap_res_dict, mask, ids_restore, ids_keep, ids_mask

    def forward_decoder(self, x, ids_restore, ids_keep, ids_mask):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        if self.cross_attn_decoder:
            # cross-attention decoding from EfficientSAM
            # Reference: https://arxiv.org/abs/2312.00863
            cls_tokens = x[:, :1, :]
            x = x[:, 1:, :]  # remove cls token

            # separate tokens into unmasked and masked groups
            unmasked_tokens = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            masked_tokens = torch.gather(x, dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, x.shape[2]))

            for blk in self.decoder_blocks:
                # query: masked_tokens, key/value: all tokens
                # note that masked_tokens in key/value would be updated through out the blocks (follow EfficientSAM)
                kv = torch.cat([cls_tokens, unmasked_tokens, masked_tokens], dim=1)
                masked_tokens = blk(masked_tokens, kv)

            x_ = torch.cat([unmasked_tokens, masked_tokens], dim=1)  # merge unmasked and masked tokens
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([cls_tokens, x_], dim=1)  # append cls token
        else:
            # original self-attention decoding
            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_teacher(self, x, ids_restore, ids_keep, ids_mask, teacher_name):
        if not self.direct_distillation:
            # embed tokens
            x = self.decoder_embed_teacher[teacher_name](x)

            # append mask tokens to sequence
            mask_tokens = self.mask_token_teacher[teacher_name].repeat(
                x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
            )
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

            # add pos embed
            x = x + self.decoder_pos_embed_teacher

            if self.cross_attn_decoder_teacher:
                # fix encoding tokens
                fix_encoding_tokens = torch.cat(
                    [
                        x[:, :1, :],
                        torch.gather(x[:, 1:, :], dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1])),
                    ],
                    dim=1,
                )
                x = torch.gather(x[:, 1:, :], dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

                # apply Transformer blocks
                for blk in self.decoder_blocks_teacher[teacher_name]:
                    x = blk(x, fix_encoding_tokens, ids_restore)

                # full sequence recovery
                x_ = torch.cat([fix_encoding_tokens[:, 1:, :], x], dim=1)
                x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1]))  # unshuffle
                x = torch.cat([fix_encoding_tokens[:, :1, :], x_], dim=1)

                # # cross-attention decoding from EfficientSAM
                # # Reference: https://arxiv.org/abs/2312.00863
                # cls_tokens = x[:, :1, :]
                # x = x[:, 1:, :]  # remove cls token

                # # separate tokens into unmasked and masked groups
                # unmasked_tokens = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]))
                # masked_tokens = torch.gather(x, dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, x.shape[2]))

                # for blk in self.decoder_blocks_teacher:
                #     # query: masked_tokens, key/value: all tokens
                #     # note that masked_tokens in key/value would be updated through out the blocks (follow EfficientSAM)
                #     kv = torch.cat(
                #         [cls_tokens, unmasked_tokens, masked_tokens], dim=1
                #     )  # TODO: check if we need to move this part into the cross-attn block for pre-normalization of the masked_tokens
                #     masked_tokens = blk(masked_tokens, kv)

                # x_ = torch.cat([unmasked_tokens, masked_tokens], dim=1)  # merge unmasked and masked tokens
                # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
                # x = torch.cat([cls_tokens, x_], dim=1)  # append cls token
            else:
                # original self-attention decoding
                # apply Transformer blocks
                for blk in self.decoder_blocks_teacher[teacher_name]:
                    x = blk(x)

            x = self.decoder_norm_teacher[teacher_name](x)

        # remove cls token
        x = x[:, 1:, :]

        # feature projection
        features = self.pred_head_teacher[teacher_name](x)

        return features

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_teacher(self, target, pred):
        """
        target: [N, L, D]
        pred: [N, L, D]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        if self.norm_pix_loss_teacher:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # TODO: check the choice of loss function
        if self.loss_type_teacher == "l2":
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        elif self.loss_type_teacher == "cos_sim":
            pred = pred / (pred.norm(dim=-1, p=2, keepdim=True) + 1.0e-6)
            target = target / (target.norm(dim=-1, p=2, keepdim=True) + 1.0e-6)
            loss = 1 - (pred * target).sum(dim=-1)  # [N, L], cosine similarity loss per patch
        else:
            raise NotImplementedError(f"Unknown loss_type_teacher: {self.loss_type_teacher}")

        return loss.mean()  # mean loss on all patches

    def forward(self, imgs, mask_ratio=0.75, teacher_features={}):
        if self.direct_distillation:
            assert mask_ratio == 0, "mask_ratio should be 0 for direct distillation"
            assert not self.reconstruct_orig_img, "reconstruct_orig_img should be False for direct distillation"
        else:
            assert mask_ratio > 0, "mask_ratio should be > 0 for image/feature reconstruction"

        latent, latent_tsap, mask, ids_restore, ids_keep, ids_mask = self.forward_encoder(imgs, mask_ratio)

        if self.reconstruct_orig_img:  # include original MAE loss
            pred = self.forward_decoder(latent, ids_restore, ids_keep, ids_mask)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, mask)
        else:
            pred = None
            loss = None

        if len(teacher_features) == 0:
            assert self.num_teachers == 0, "Please provide teacher_features for feature reconstruction"
            assert self.reconstruct_orig_img, "Either provide teacher_names or set reconstruct_orig_img to True"
            return loss, None, pred, mask  # no teacher

        # feature reconstruction
        # pred_teacher_features = self.forward_decoder_teacher(latent, ids_restore, ids_keep, ids_mask)
        pred_teacher_features = {}
        for teacher_name in self.teacher_names:
            pred_teacher_features[teacher_name] = self.forward_decoder_teacher(
                latent_tsap[teacher_name], ids_restore, ids_keep, ids_mask, teacher_name
            )

        # teacher loss
        loss_teacher_dict = {}
        for idx, teacher_name in enumerate(self.teacher_names):
            loss_teacher_dict[teacher_name] = {}
            loss_teacher = self.forward_loss_teacher(
                teacher_features[teacher_name],
                pred_teacher_features[teacher_name],
            )
            loss_teacher_dict[teacher_name]["patch"] = loss_teacher * self.loss_weights_teacher[teacher_name]
            loss_teacher_dict[teacher_name]["load_balance"] = 0.0  # deprecated

        loss_teacher_balance = None  # deprecated

        return loss, loss_teacher_dict, loss_teacher_balance, pred, mask


def mim_vit_small_patch16(**kwargs):
    model = MaskedImageModelingViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mim_vit_base_patch16(**kwargs):
    model = MaskedImageModelingViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mim_vit_large_patch16(**kwargs):
    model = MaskedImageModelingViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mim_vit_huge_patch14(**kwargs):
    model = MaskedImageModelingViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
