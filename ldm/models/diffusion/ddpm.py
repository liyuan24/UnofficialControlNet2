from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange

from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import default, instantiate_from_config


def disabled_train(self, mode=True):
    return self


class DiffusionWrapper:
    pass


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
        self.num_timesteps = timesteps

    def register_schedule(self, timesteps=1000, linear_start=1e-4, linear_end=2e-2):
        betas = make_beta_schedule(timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1. - betas
        alpha_cumprod = np.cumprod(alphas, axis=0)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alpha_cumprod', to_torch(alpha_cumprod))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alpha_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alpha_cumprod)))

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # forward process noise
        q_noise = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.model(q_noise, t)

        loss_dict = {}
        target = q_noise
        loss = self.get_loss(model_output, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean()
        return loss_simple, loss_dict

    def get_loss(self, pred, target, mean=True):
        if mean:
            loss = torch.nn.MSELoss()
        else:
            loss = torch.nn.MSELoss(reduction='none')
        return loss(pred, target)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        # k: the key of the dict of batch
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        self.log_dict(loss_dict_no_ema, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""

    def __init__(
            self,
            first_stage_config,  # image -> latent images
            cond_stage_config,  # prompt
            cond_stage_key="txt",
            cond_stage_trainable=False,
            conditioning_key='crossattn',  # crosattn
            scale_factor=0.1825,
            *args, **kwargs
    ):
        super().__init__(conditioning_key, *args, **kwargs)
        self.cond_stage_trainable = cond_stage_trainable
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.scale_factor = scale_factor

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        if self.cond_stage_trainable:
            self.cond_stage_model = model
        else:
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type:'{type(encoder_posterior)}' not implemented yet")
        return self.scale_factor * z

    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, return_x=False):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(device=self.device)

