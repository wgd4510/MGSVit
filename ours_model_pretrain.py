# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on the MAE code base
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import torch
from torch import nn, einsum

from einops import rearrange
from models.transformer_model import get_2d_sincos_pos_embed
from models.transformer_model import BaseTransformer, BaseT_shortcut


class base_encoder(nn.Module):
    def __init__(
            self, 
            in_dim, 
            num_patches=196, 
            encoder_dim=768, 
            encoder_depth=12, 
            encoder_num_heads=16, 
            shortnum=4):
        super().__init__()
        self.base_encoder = BaseT_shortcut(dim=encoder_dim, depth=encoder_depth, num_heads=encoder_num_heads, shortnum=shortnum)

        # Setup position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_dim), requires_grad=False)  # fixed sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        self.linear_input = nn.Linear(in_dim, encoder_dim)


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
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward(self, x, mask_ratio=0.75):  # (bs, 196, 512)  (bs, 196, 3072)
        x = self.linear_input(x)   # (bs, 196, 768)
        x = x + self.pos_embed  # (bs, seq, encoder_dim)  (bs, 196, 768)
        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
            x = self.base_encoder(x)  # (bs, 196, 768)
            return x, mask, ids_restore
        else:
            x = self.base_encoder(x)  # (bs, 196, 768)
            return x
    

class MCMAE(nn.Module):
    def __init__(
            self, 
            in_dim,
            out_dim, 
            base_encoder,
            num_patches=196, 
            encoder_dim=768, 
            encoder_depth=12, 
            encoder_num_heads=12, 
            decoder_dim=384, 
            decoder_depth=2, 
            decoder_num_heads=6, 
            shortnum=4,
            moco_dim=256,
            moco_mlp_dim=4096,
            T=1.0
            ):
        super().__init__()
        # ============= build base encoder ============= 
        self.encoder = base_encoder(in_dim, 
                                    num_patches, 
                                    encoder_dim, 
                                    encoder_depth, 
                                    encoder_num_heads, 
                                    shortnum)
        
        self.momentum_encoder = base_encoder(in_dim, 
                                             num_patches, 
                                             encoder_dim, 
                                             encoder_depth, 
                                             encoder_num_heads, 
                                             shortnum)
        
        self.decoder = BaseTransformer(dim=decoder_dim, 
                                       depth=decoder_depth, 
                                       num_heads=decoder_num_heads)

        # ============= build MAE layer =============
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim),
            requires_grad=False)  # fixed sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 
                                                    int(num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.linear_output = nn.Linear(decoder_dim, out_dim)

        # ============= build MoCo layer =============
        self.T = T
        # projectors
        self.encoder_head = self._build_mlp(3, encoder_dim, moco_mlp_dim, moco_dim)
        self.momentum_encoder_head = self._build_mlp(3, encoder_dim, moco_mlp_dim, moco_dim)
        # predictor
        self.predictor = self._build_mlp(2, moco_dim, moco_mlp_dim, moco_dim)

        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        for param_b, param_m in zip(self.encoder_head.parameters(), self.momentum_encoder_head.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    # --------------------------------------------------------------------------
    def forward_decoder(self, x_in, ids_restore):
        # embed tokens
        x = self.enc_to_dec(x_in)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(
                1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = self.decoder(x)

        # predictor projection
        return self.linear_output(x)
    

    def autoencoder_loss(self, imgs, pred, mask):
        loss = (pred - imgs)**2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    

    # --------------------------------------------------------------------------
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
        for param_b, param_m in zip(self.encoder_head.parameters(), self.momentum_encoder_head.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output


    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = self.concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        # labels = (torch.arange(N, dtype=torch.long)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)


    # --------------------------------------------------------------------------
    def forward(self, x1, x2, m, mask_ratio=0.75):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
            mask_ratio: mae mask ratio
        Output:
            loss = loss_mae + loss_mocov3
        """
        # 先对两个输入数据加mask掩膜， 并进行掩膜后的encode编码
        mae_x1, mask_x1, ids_restore_x1 = self.encoder(x1, mask_ratio)
        mae_x2, mask_x2, ids_restore_x2 = self.encoder(x2, mask_ratio)

        # 对掩膜后特征进行自编码重构回原图
        pred_x1 = self.forward_decoder(mae_x1, ids_restore_x1)
        pred_x2 = self.forward_decoder(mae_x2, ids_restore_x2)

        # 计算 MAE 的自编码重构损失
        loss_mae = self.autoencoder_loss(x1, pred_x1, mask_x1) + self.autoencoder_loss(x2, pred_x2, mask_x2)

        # 对原始两个输入数据分别进行encode编码 
        moco_x1 = self.encoder(x1, mask_ratio=0)
        moco_x2 = self.encoder(x2, mask_ratio=0)
        q1 = self.predictor(self.encoder_head(moco_x1.mean(dim=1)))
        q2 = self.predictor(self.encoder_head(moco_x2.mean(dim=1)))

        # 更新momentum encoder 并进行编码
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1, mask_ratio=0)
            k1 = self.momentum_encoder_head(k1.mean(dim=1))  # [bs, 768] 
            k2 = self.momentum_encoder(x2, mask_ratio=0)
            k2 = self.momentum_encoder_head(k2.mean(dim=1))  # [bs, 768] 

        # 计算 q1和k2 q2和k1 的对比损失
        loss_moco = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        return loss_mae + loss_moco
