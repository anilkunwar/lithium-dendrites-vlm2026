"""Conditional DDPM: (simulation parameters, physical time) -> stacked RGB images of eta, c, pot.

The condition embedding (parameter MLP + Fourier features of the normalized time) is added
to the diffusion-timestep embedding through UNet2DModel's identity class embedding.
With cond_dropout, a learned null embedding allows classifier-free guidance.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from diffusers import DDIMScheduler, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class FourierTime(nn.Module):
    """s in [0, 1] -> [s, sin(2^k pi s), cos(2^k pi s)] for k < num_freqs."""

    def __init__(self, num_freqs=8):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(num_freqs), persistent=False)
        self.dim = 1 + 2 * num_freqs

    def forward(self, s):
        a = s[:, None] * self.freqs
        return torch.cat([s[:, None], a.sin(), a.cos()], dim=1)


class ParamDiffusion(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_channels: int = 9,
        num_params: int = 14,
        sample_size: int = 128,
        block_out_channels: tuple = (64, 128, 256, 256),
        layers_per_block: int = 2,
        attention_levels: tuple = (False, False, True, True),
        time_freqs: int = 8,
        cond_dropout: float = 0.0,
    ):
        super().__init__()
        down = tuple("AttnDownBlock2D" if a else "DownBlock2D" for a in attention_levels)
        up = tuple("AttnUpBlock2D" if a else "UpBlock2D" for a in reversed(attention_levels))
        emb_dim = block_out_channels[0] * 4
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=num_channels,
            out_channels=num_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            down_block_types=down,
            up_block_types=up,
            class_embed_type="identity",
        )
        self.time_features = FourierTime(time_freqs)
        self.cond_mlp = nn.Sequential(
            nn.Linear(num_params + self.time_features.dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.null_cond = nn.Parameter(torch.zeros(emb_dim)) if cond_dropout > 0 else None

    def embed(self, params, time, drop=None):
        """params (B, P) z-scores, time (B,) normalized; drop (B,) bool uses the null embedding."""
        e = self.cond_mlp(torch.cat([params, self.time_features(time)], dim=1))
        if drop is not None:
            e = torch.where(drop[:, None], self.null_cond.to(e.dtype), e)
        return e

    def forward(self, noisy, timestep, params, time, drop=None):
        return self.unet(noisy, timestep, class_labels=self.embed(params, time, drop)).sample


@torch.no_grad()
def sample(model, scheduler: DDIMScheduler, params, time, num_steps=50, guidance=1.0, generator=None):
    """Draw fields in normalized space for normalized conditions params (B, P), time (B,).

    guidance != 1: classifier-free guidance, out = out_uncond + guidance * (out_cond - out_uncond).
    """
    use_cfg = guidance != 1.0
    if use_cfg and model.null_cond is None:
        raise ValueError("guidance != 1 needs a model trained with --cond-dropout > 0")
    cfg = model.config
    device, dtype = model.device, next(model.parameters()).dtype
    params, time = params.to(device, dtype), time.to(device, dtype)
    B = params.shape[0]
    x = torch.randn((B, cfg.num_channels, cfg.sample_size, cfg.sample_size),
                    generator=generator, device=device, dtype=dtype)
    scheduler.set_timesteps(num_steps, device=device)
    for t in scheduler.timesteps:
        tt = t.expand(B)
        if use_cfg:
            drop = torch.cat([torch.zeros(B, dtype=torch.bool, device=device),
                              torch.ones(B, dtype=torch.bool, device=device)])
            out = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([params, params]),
                        torch.cat([time, time]), drop)
            cond, uncond = out.chunk(2)
            out = uncond + guidance * (cond - uncond)
        else:
            out = model(x, tt, params, time)
        x = scheduler.step(out, t, x, generator=generator).prev_sample
    return x
