import torch
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn.functional as F

from ldm.util import default, exist

import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def zero_module(module):
    """
    Zero out the parameters of a module and return it
    """
    # detach will detach the parameters from the computation graph and set the requires_grad = False
    for p in module.parameters():
        p.detach().zero()
    return module


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = default(context_dim, inner_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        h = self.heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force to float32 to prevent overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exist(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        sim = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj_in = nn.Linear(dim_in, 2 * dim_out)

    def forward(self, x):
        x, gate = self.proj_in(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        if glu:
            project_in = GEGLU(dim, inner_dim)
        else:
            project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        attn_module = CrossAttention
        self.attn1 = attn_module(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head,
                                 dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_module(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head,
                                 dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context) + x
        x = self.attn2(self.norm2(x), context) + x
        return self.ff(self.norm3(x)) + x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data
    First, project the input and reshape to b, t, d
    Then apply standard transformer action.
    Finally, reshape back to image
    NEW: use_linear for more efficiency instead of the 1x1 convolutions
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if context_dim is not None and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalize(in_channels)
        if use_linear:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout,
                                   context_dim=context_dim[d],
                                   checkpoint=use_checkpoint) for d in range(
                depth)])
        if use_linear:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        else:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        # only after change the shape, we can use linear project here
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context[i])
        # if use linear, before rearrange the shape, use linear projection
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out
        return x + x_in

