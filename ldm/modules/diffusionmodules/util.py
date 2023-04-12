import torch


def make_beta_schedule(timesteps, linear_start=4e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, timesteps, dtype=torch.float32) ** 2
    return betas.numpy()
