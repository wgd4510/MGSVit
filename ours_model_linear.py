# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on the MAE code base
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

import torch
from torch import nn, einsum
from einops import rearrange
from models.transformer_model import get_2d_sincos_pos_embed
from models.transformer_model import BaseT_shortcut


class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=True)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=True)
        self.hidden_sigmoid = nn.Linear(size_out * 2, 1, bias=True)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))

        return z.view(z.size()[0], 1) * h1 + (1-z).view(z.size()[0], 1) * h2
    

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
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        self.linear_input = nn.Linear(in_dim, encoder_dim)


    def forward(self, x):  # (bs, 196, 512)  (bs, 196, 3072)
        x = self.linear_input(x)   # (bs, 196, 768)
        x = x + self.pos_embed  # (bs, seq, encoder_dim)  (bs, 196, 768)
        x = self.base_encoder(x)  # (bs, 196, 768)
        return x
    

def mcmae_v1(pretrained, **kwargs):
    model = base_encoder(**kwargs)
    if pretrained:
        checkpoint= torch.load(pretrained, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrained)
        checkpoint_model = checkpoint['model']
        dict_model = {}
        for k, v in checkpoint_model.items():
            if k[:8] == 'encoder.':
                dict_model[k[8:]] = v
        msg = model.load_state_dict(dict_model, strict=False)
        print(msg)
    return model


class MCMAE_Encoder_single(nn.Module):
    def __init__(
            self, 
            io_dim,
            weights,
            num_patches=196, 
            encoder_dim=768, 
            encoder_depth=12, 
            encoder_num_heads=16, 
            shortnum=4,
            num_classes=19
            ):
        super().__init__()
        self.backbone = mcmae_v1(weights, in_dim=io_dim, num_patches=num_patches, encoder_dim=encoder_dim, encoder_depth=encoder_depth, encoder_num_heads=encoder_num_heads, shortnum=shortnum)

        self.head = nn.Linear(encoder_dim, num_classes) if num_classes > 0 else nn.Identity()

    
    def forward(self, encodings):  # (bs, 196, 512)  (bs, 196, 3072)
        x = self.backbone(encodings)
        x = x.mean(dim=1)  # [bs, 768] 
        x = self.head(x)
        return x
    

class MCMAE_Encoder(nn.Module):
    def __init__(
            self, 
            io_dim_s1,
            io_dim_s2,
            S1_weights,
            S2_weights,
            num_patches=196, 
            encoder_dim=768, 
            encoder_depth=12, 
            encoder_num_heads=16, 
            shortnum=4,
            num_classes=19
            ):
        super().__init__()

        self.S1 = mcmae_v1(S1_weights, in_dim=io_dim_s1, num_patches=num_patches, encoder_dim=encoder_dim, encoder_depth=encoder_depth, encoder_num_heads=encoder_num_heads, shortnum=shortnum)
        self.S2 = mcmae_v1(S2_weights, in_dim=io_dim_s2, num_patches=num_patches, encoder_dim=encoder_dim, encoder_depth=encoder_depth, encoder_num_heads=encoder_num_heads, shortnum=shortnum)

        self.head = nn.Linear(encoder_dim, num_classes) if num_classes > 0 else nn.Identity()
   
        self.gmu = GatedMultimodalLayer(size_in1=encoder_dim, size_in2=encoder_dim, size_out=encoder_dim)

    def forward_repo(self, s1_encodings, s2_encodings):  # (bs, 196, 512)  (bs, 196, 3072)
        x_s1 = self.S1(s1_encodings)
        x_s2 = self.S2(s2_encodings)
        x_s1 = x_s1.mean(dim=1)  # [bs, 768] 
        x_s2 = x_s2.mean(dim=1)  # [bs, 768] 
        x = self.gmu(x_s1, x_s2)  # (bs, 768)
        x = self.head(x)
        return x
    
    def forward(self, s1_encodings, s2_encodings):
        x = self.forward_repo(s1_encodings, s2_encodings)
        return x
