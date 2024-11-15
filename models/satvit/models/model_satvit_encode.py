import torch
from torch import nn
from torch import nn, einsum
from einops import rearrange
import math
from models.transformer_model import get_2d_sincos_pos_embed
from models.transformer_model import BaseTransformer, BaseT_shortgate_C3, BaseT_shortgate_Cn

# --------------------------------------------------------
# Based on the MAE code base
# https://github.com/facebookresearch/mae
# --------------------------------------------------------


class SatViT_Encode(nn.Module):
    def __init__(self, in_dim, num_patches=256, encoder_dim=768, encoder_depth=12, encoder_num_heads=16, num_classes=19, head_depth=1):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_patches = num_patches

        self.encoder = BaseTransformer(
            dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
        )

        # Setup position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, encoder_dim),
            requires_grad=False) 
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Input and output maps
        self.linear_input = nn.Linear(in_dim, encoder_dim)

        if head_depth == 1:
            self.head = nn.Linear(self.encoder_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            hidden_dim1 = int(self.encoder_dim / 2)
            hidden_dim2 = int(self.encoder_dim / 4)
            self.head = nn.Sequential(
                nn.Linear(self.encoder_dim, hidden_dim1, bias=True),
                nn.ReLU(True),
                nn.Linear(hidden_dim1, hidden_dim2, bias=True),
                nn.ReLU(True),
                nn.Linear(hidden_dim2, num_classes, bias=True),
            )

    # 原始的forward
    def forward_repo(self, patch_encodings):
        x = self.linear_input(patch_encodings)   # (bs, 196, 768)
        x = x + self.pos_embed  # (bs, seq, encoder_dim)  (bs, 196, 768)
        x = self.encoder(x)  # (bs, 196, 768)
        x = x.mean(dim=1)  # [bs, 768] 
        x = self.head(x)
        return x

    def forward(self, patch_encodings):
        x = self.forward_repo(patch_encodings)
        return x


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
    

class SatViT_Encode_multimodel(nn.Module):
    def __init__(self, in_dim_s1, in_dim_s2, num_patches=256, encoder_dim=768, encoder_depth=12, encoder_num_heads=16, num_classes=19, head_depth=1, shortnum=0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_patches = num_patches
        if shortnum == 0:
            self.encoder1 = BaseTransformer(
                dim=encoder_dim,
                depth=encoder_depth,
                num_heads=encoder_num_heads,
            )
            self.encoder2 = BaseTransformer(
                dim=encoder_dim,
                depth=encoder_depth,
                num_heads=encoder_num_heads,
            )
        else:
            self.encoder1 = BaseT_shortgate_C3(
                dim=encoder_dim,
                depth=encoder_depth,
                num_heads=encoder_num_heads,
                shortnum=shortnum
            )
            self.encoder2 = BaseT_shortgate_C3(
                dim=encoder_dim,
                depth=encoder_depth,
                num_heads=encoder_num_heads,
                shortnum=shortnum
            )
        # Setup position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, encoder_dim),
            requires_grad=False) 
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Input and output maps
        self.linear_input1 = nn.Linear(in_dim_s1, encoder_dim)
        self.linear_input2 = nn.Linear(in_dim_s2, encoder_dim)

        if head_depth == 1:
            self.head = nn.Linear(self.encoder_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            hidden_dim1 = int(self.encoder_dim / 2)
            hidden_dim2 = int(self.encoder_dim / 4)
            self.head = nn.Sequential(
                nn.Linear(self.encoder_dim, hidden_dim1, bias=True),
                nn.ReLU(True),
                nn.Linear(hidden_dim1, hidden_dim2, bias=True),
                nn.ReLU(True),
                nn.Linear(hidden_dim2, num_classes, bias=True),
            )

        self.gmu = GatedMultimodalLayer(size_in1=encoder_dim, size_in2=encoder_dim, size_out=encoder_dim)

    # 原始的forward
    def forward_repo(self, s1_encodings, s2_encodings):  # (bs, 196, 512)  (bs, 196, 3072)
        x_s1 = self.linear_input1(s1_encodings)   # (bs, 196, 768)
        x_s2 = self.linear_input2(s2_encodings)   # (bs, 196, 768)
        x_s1 = x_s1 + self.pos_embed  # (bs, seq, encoder_dim)  (bs, 196, 768)
        x_s2 = x_s2 + self.pos_embed  # (bs, seq, encoder_dim)  (bs, 196, 768)
        x_s1 = self.encoder1(x_s1)  # (bs, 196, 768)
        x_s2 = self.encoder2(x_s2)  # (bs, 196, 768)
        x_s1 = x_s1.mean(dim=1)  # [bs, 768] 
        x_s2 = x_s2.mean(dim=1)  # [bs, 768] 
        x = self.gmu(x_s1, x_s2)  # (bs, 768)
        x = self.head(x)
        return x

    def forward(self, s1_encodings, s2_encodings):
        x = self.forward_repo(s1_encodings, s2_encodings)
        return x


class SatViT_Encode_multimodel_multiloss(SatViT_Encode_multimodel):
    def __init__(self, in_dim_s1, in_dim_s2, num_patches=256, encoder_dim=768, encoder_depth=12, encoder_num_heads=16, num_classes=19, head_depth=1, shortnum=0):
        super().__init__(in_dim_s1, in_dim_s2, num_patches, encoder_dim, encoder_depth, encoder_num_heads, num_classes, head_depth, shortnum)
        assert shortnum > 0
        self.encoder1 = BaseT_shortgate_Cn(
            dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            shortnum=shortnum
        )
        self.encoder2 = BaseT_shortgate_Cn(
            dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            shortnum=shortnum
        )
        self.layers = nn.ModuleList([])
        self.loss_num = int(encoder_depth / shortnum)
        for i in range(self.loss_num):
            self.layers.append(
                nn.ModuleList([
                    GatedMultimodalLayer(size_in1=encoder_dim, size_in2=encoder_dim, size_out=encoder_dim),
                    nn.Linear(self.encoder_dim, num_classes) if num_classes > 0 else nn.Identity(),
                ]))

    def forward(self, s1_encodings, s2_encodings):  # (bs, 196, 512)  (bs, 196, 3072)
        out = []
        x_s1 = self.linear_input1(s1_encodings)   # (bs, 196, 768)
        x_s2 = self.linear_input2(s2_encodings)   # (bs, 196, 768)
        x_s1 = x_s1 + self.pos_embed  # (bs, seq, encoder_dim)  (bs, 196, 768)
        x_s2 = x_s2 + self.pos_embed  # (bs, seq, encoder_dim)  (bs, 196, 768)
        hiddens_out_s1, out_s1 = self.encoder1(x_s1)  # (loss_num, bs, 196, 768)
        hiddens_out_s2, out_s2 = self.encoder2(x_s2)  # (loss_num, bs, 196, 768)
        for i, (l_gmu, l_head) in enumerate(self.layers):
            x_s1 = hiddens_out_s1[i, :, :, :]  # [bs, 196, 768] 
            x_s2 = hiddens_out_s2[i, :, :, :]  # [bs, 196, 768] 
            x_s1 = x_s1.mean(dim=1)  # [bs, 768] 
            x_s2 = x_s2.mean(dim=1)  # [bs, 768] 
            x = l_gmu(x_s1, x_s2)  # (bs, 768)
            x = l_head(x)  # (bs, num_classes)
            out.append(x)
        hidden_out = torch.stack(out)  # (loss_num, bs, num_classes)
        x_s1 = out_s1.mean(dim=1)  # [bs, 768] 
        x_s2 = out_s2.mean(dim=1)  # [bs, 768] 
        x = self.gmu(x_s1, x_s2)  # (bs, 768)
        x = self.head(x)
        return hidden_out, x


# Encode结构最后添加了gate门控单元，经测试效果不好
class SatViT_Encode_Twogate(SatViT_Encode):
    def __init__(self, gate_num=2, **kwargs):
        super().__init__(**kwargs)

        self.encoder_dim = kwargs['encoder_dim']
        self.num_patches = kwargs['num_patches']
        self.gate_num = gate_num
     
        # 多头sigmoid门控
        if self.gate_num != 0:
            self.size_in = int(self.num_patches / gate_num)
            self.size_out = 1
            self.hidden = nn.Linear(self.size_in, self.size_out, bias=False)
            self.hidden_sigmoid = nn.Linear(self.encoder_dim * 2, 1, bias=False)
            self.tanh_f = nn.Tanh()
            self.sigmoid_f = nn.Sigmoid()


    # 双头sigmoid门控
    def double_gate(self, x):
        x = x.transpose(1, 2)  # (bs, 768, 196)
        x1 = x[:, :, :self.size_in]  # (bs, 768, 98)
        x2 = x[:, :, self.size_in:]  # (bs, 768, 98)
        h1 = self.tanh_f(self.hidden(x1).squeeze())  # (bs, 768)
        h2 = self.tanh_f(self.hidden(x2).squeeze())  # (bs, 768)
        x = torch.cat((h1, h2), dim=1)  # (bs, 2 * 786)
        z = self.sigmoid_f(self.hidden_sigmoid(x))  # (bs, 1)
        out = z.view(z.size()[0], 1) * h1 + (1 - z).view(z.size()[0], 1) * h2  # (bs, 768)
        return out

    # 四头sigmoid门控
    def four_gate(self, x):
        x = x.transpose(1, 2)  # (bs, 768, 196)
        q, k, v, w = x.chunk(4, dim=2)  # (bs, 768, 49)
        h1 = self.tanh_f(self.hidden(q).squeeze())  # (bs, 768)
        h2 = self.tanh_f(self.hidden(k).squeeze())  # (bs, 768)
        h3 = self.tanh_f(self.hidden(v).squeeze())  # (bs, 768)
        h4 = self.tanh_f(self.hidden(w).squeeze())  # (bs, 768)
        x1 = torch.cat((h1, h2), dim=1)  # (bs, 2 * 786)
        x2 = torch.cat((h3, h4), dim=1)  # (bs, 2 * 786)
        z1 = self.sigmoid_f(self.hidden_sigmoid(x1))  # (bs, 1)
        z2 = self.sigmoid_f(self.hidden_sigmoid(x2))  # (bs, 1)
        out1 = z1.view(z1.size()[0], 1) * h1 + (1 - z1).view(z1.size()[0], 1) * h2  # (bs, 768)
        out2 = z2.view(z2.size()[0], 1) * h3 + (1 - z2).view(z2.size()[0], 1) * h4  # (bs, 768)
        out = torch.stack((out1, out2), dim=1)  # (bs, 2, 786)
        return out

    # 双头sigmoid门控
    def forward_doublegate(self, patch_encodings):
        x = self.linear_input(patch_encodings)   # (bs, 196, 768)
        x = x + self.pos_embed  # (bs, seq, encoder_dim)  (bs, 196, 768)
        x = self.encoder(x)  # (bs, 196, 768)
        x = self.double_gate(x)
        x = self.head(x)
        return x
        
    # 四头sigmoid门控
    def forward_fourgate(self, patch_encodings):
        x = self.linear_input(patch_encodings)   # (bs, 196, 768)
        x = x + self.pos_embed  # (bs, seq, encoder_dim)  (bs, 196, 768)
        x = self.encoder(x)  # (bs, 196, 768)
        x = self.four_gate(x)
        x = x.mean(dim=1)  # [bs, 768] 
        x = self.head(x)
        return x


    def forward(self, patch_encodings):
        if self.gate_num == 2:
            x = self.forward_doublegate(patch_encodings)
        elif self.gate_num == 4:
            x = self.forward_fourgate(patch_encodings)
        return x


# BaseT结构内部添加了gate门控单元，Encode结构最后添加了Attention自注意力结构，经测试效果不好
class SatViT_Encode_Attn(SatViT_Encode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 自注意力门控(经测试精度下降5个点)
        self.encoder_dim = kwargs['encoder_dim']
        self.num_patches = kwargs['num_patches']
        self.encoder_num_heads = kwargs['encoder_num_heads']
        self.encoder_depth = kwargs['encoder_depth']

        self.num_header = self.encoder_num_heads
        self.step_num = int(math.log(self.num_patches, 3))
        dim_head = int(self.encoder_dim / self.encoder_num_heads)
        self.scale = dim_head**-0.5
        self.dropout = nn.Dropout(0.)

        from models.transformer_model import BaseTransformer_gate
        self.encoder = BaseTransformer_gate(
            dim=self.encoder_dim,
            depth=self.encoder_depth,
            num_heads=self.encoder_num_heads,
        )

    # 自注意力门控(经测试精度下降5个点)
    def attention_gate(self, x):
        q, k, v = x[:,-int(x.shape[1] / 3) * 3:,:].chunk(3, dim=1)  # 3 * [bs, 65, 768]
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_header),
                      (q, k, v))  # (BSZ, num_heads, num_patches, dim_head)  [bs, 12, 65, 64]
        attention_scores = einsum(
            'b h i d, b h j d -> b h i j', q,
            k) * self.scale  # (BSZ, num_heads, num_patches, num_patches) [bs, 12, 65, 65]
        attn = attention_scores.softmax(
            dim=-1)  # (BSZ, num_heads, num_patches, num_patches)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)  [bs, 12, 65, 65]
        out = einsum('b h i j, b h j d -> b h i d', attn,
                     v)  # (BSZ, num_heads, num_patches, dim_head)  [bs, 12, 65, 64]
        out = rearrange(out, 'b h n d -> b n (h d)')  # (BSZ, num_patches, dim)  [bs, 65, 768]
        return out
    

    # 添加了自注意力门控的forward
    def forward(self, patch_encodings):
        x = self.linear_input(patch_encodings)   # (bs, 196, 768)
        x = x + self.pos_embed  # (bs, seq, encoder_dim)  (bs, 196, 768)
        # 添加门控gate的BaseT结构
        x = self.encoder(x)  # (bs, 196, 768)
        # 添加自注意力Attn结构
        for _ in range(self.step_num):
            x = self.attention_gate(x)  # [bs, 65, 768] [bs, 21, 768] [bs, 7, 768] [bs, 2, 768] 

        x = x.mean(dim=1)  # [bs, 768] 
        x = self.head(x)
        return x

