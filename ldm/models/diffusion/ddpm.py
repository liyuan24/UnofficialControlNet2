from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch

from ldm.modules.diffusionmodules.util import make_beta_schedule
from ldm.util import default


class DDPM(pl.LightningModule):
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 conditioning_key=None,
                 ):
        super.__init__()
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.channels = channels
        self.image_size = image_size
        self.register_schedule(timesteps=timesteps, linear_start=linear_start, linear_end=linear_end)
        self.first_stage_key = first_stage_key
        self.log_every_t = log_every_t

    def register_schedule(self, timesteps=1000, linear_start=1e-4, linear_end=2e-2):
        betas = make_beta_schedule(timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1. - betas
        alpha_cumprod = np.cumprod(alphas, axis=0)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alpha_cumprod', to_torch(alpha_cumprod))

    def p_loss(self, x_start, t, noise=None):
        noise = default(noise, torch.randn_like(x_start))



