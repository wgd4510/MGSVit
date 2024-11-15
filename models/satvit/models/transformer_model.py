import torch
from torch import nn, einsum
from einops import rearrange
import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                   axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class FFN(nn.Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.,
    ):
        """
        FFN (FeedForward Network)
        :param dim: model dimension (number of features)
        :param mult: multiply the model dimension by mult to get the FFN's inner dimension
        :param dropout: dropout between 0 and 1
        """
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),  # (BSZ, num_patches, inner_dim)
            nn.GELU(),  # (BSZ, num_patches, inner_dim)
            nn.Dropout(dropout),  # (BSZ, num_patches, inner_dim)
            nn.Linear(inner_dim, dim)  # (BSZ, num_patches, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        return self.net(x)  # (BSZ, num_patches, dim)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            dropout=0.,
    ):
        """
        Self-Attention module
        :param dim: model dimension (number of features)
        :param num_heads: number of attention heads
        :param dropout: dropout between 0 and 1
        """
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, 'dim must be evenly divisible by num_heads'
        dim_head = int(dim / num_heads)
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # (BSZ, num_patches, dim)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_heads),
                      (q, k, v))  # (BSZ, num_heads, num_patches, dim_head)

        attention_scores = einsum(
            'b h i d, b h j d -> b h i j', q,
            k) * self.scale  # (BSZ, num_heads, num_patches, num_patches)

        attn = attention_scores.softmax(
            dim=-1)  # (BSZ, num_heads, num_patches, num_patches)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        out = einsum('b h i j, b h j d -> b h i d', attn,
                     v)  # (BSZ, num_heads, num_patches, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


class BaseTransformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(
                        dim=dim, num_heads=num_heads, dropout=attn_dropout),
                    FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                ]))

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x):
        for self_attn, ffn in self.layers:
            x = self_attn(x) + x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)

        return self.norm_out(x)  # (BSZ, num_patches, dim)


# 经过测试，加了门控gate的模型结果效果不好
class BaseTransformer_gate(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(
                        dim=dim, num_heads=num_heads, dropout=attn_dropout),
                    FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                ]))
            
        
        self.norm_out = nn.LayerNorm(dim)

        self.hidden_f = nn.Linear(dim, 1, bias=False)
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x):
        for self_attn, ffn in self.layers:
            z = self.sigmoid_f(self.hidden_f(x))
            x = self_attn(x) + x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)
            x = x * z
        return self.norm_out(x)  # (BSZ, num_patches, dim)


class Bottleneck(nn.Module):
    def __init__(self, dim, attn_depth=3, num_heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(attn_depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(
                        dim=dim, num_heads=num_heads, dropout=attn_dropout),
                    FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                ]))
        self.hidden_f = nn.Linear(dim, 1, bias=False)
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x):
        z = x * self.sigmoid_f(self.hidden_f(x))
        for i, (self_attn, ffn) in enumerate(self.layers):
            x = self_attn(x) + x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)
        return x + z
    

# 经测试，效果有一点点提升
class BaseT_shortgate_C3(nn.Module):
    def __init__(self, dim, depth, num_heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4, shortnum=3):
        super().__init__()
        n = int(depth / shortnum)
        assert depth % shortnum == 0
        self.m = nn.Sequential(*[Bottleneck(dim, shortnum, num_heads, attn_dropout, ff_dropout, ff_mult) for _ in range(n)])
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm_out(self.m(x))  # (BSZ, num_patches, dim) 


class BaseT_shortgate_Cn(nn.Module):
    def __init__(self, dim, depth, num_heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4, shortnum=3):
        super().__init__()
        n = int(depth / shortnum)
        assert depth % shortnum == 0
        self.m = nn.Sequential(*[Bottleneck(dim, shortnum, num_heads, attn_dropout, ff_dropout, ff_mult) for _ in range(n)])
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x):
        hidden_out = []
        for layer in self.m:
            x = layer(x)
            hidden_out.append(x)
        out = self.norm_out(x)
        hiddens_out = torch.stack(hidden_out) 
        return hiddens_out, out  # (n, bs, num_patches, dim), (bs, num_patches, dim) 
    

# 经测试，效果不好
class BaseT_shortgate(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            shortnum=3
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.shortlist = []
        for i in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(
                        dim=dim, num_heads=num_heads, dropout=attn_dropout),
                    FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                ]))
            
            if i % shortnum == 0:
                self.shortlist.append(i)
        
        self.norm_out = nn.LayerNorm(dim)

        self.hidden_f = nn.Linear(dim, 1, bias=False)
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x):       
        for i, (self_attn, ffn) in enumerate(self.layers):
            if i in self.shortlist:
                z = self.sigmoid_f(self.hidden_f(x))
            x = self_attn(x) + x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)
            if i in self.shortlist:
                x = x * z
        return self.norm_out(x)  # (BSZ, num_patches, dim)


