import math

import torch
from torch import nn


def make_beta_schedule(timesteps, linear_start=4e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, timesteps, dtype=torch.float32) ** 2
    return betas.numpy()


def extract_into_tensor(a, t, x_shape):
    # a: beta, alpha and etc schedule, 1d tensor with timesteps length
    # t: selected timesteps, 1d tensor with batch size length
    # x_shape: the shape of the image
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    # timesteps[:, None] is [b, 1], freqs[None] is [1, half]
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


def group_norm(channels):
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

if __name__ == '__main__':
    a = torch.randn(10)
    t = torch.randint(10, (3,))
    x_shape = [3, 3, 3, 3]
    print(extract_into_tensor(a, t, x_shape))
