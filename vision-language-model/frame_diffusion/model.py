"""Conditional video-prediction diffusion model built from HuggingFace diffusers parts.

Context frames are concatenated with the noisy target along channels (as in MCVD /
SR3-style conditioning). The time gaps between frames and the simulation parameters are
embedded and added to the diffusion-timestep embedding through UNet2DModel's class embedding.
"""

from __future__ import annotations

import torch
from torch import nn
from diffusers import DDIMScheduler, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class FramePredictor(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_channels: int = 3,
        num_context: int = 4,
        sample_size: int = 256,
        block_out_channels: tuple = (64, 128, 256, 256, 512),
        layers_per_block: int = 2,
        attention_levels: tuple = (False, False, False, True, True),
        num_params: int = 0,
    ):
        super().__init__()
        down = tuple("AttnDownBlock2D" if a else "DownBlock2D" for a in attention_levels)
        up = tuple("AttnUpBlock2D" if a else "UpBlock2D" for a in reversed(attention_levels))
        emb_dim = block_out_channels[0] * 4
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=num_channels * (num_context + 1),
            out_channels=num_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            down_block_types=down,
            up_block_types=up,
            class_embed_type="identity",
        )
        # K log time gaps: K-1 between context frames + 1 from the last context to the target.
        self.gap_embedding = nn.Sequential(
            nn.Linear(num_context, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        # Normalized simulation parameters (z-scores), constant within a case.
        self.param_embedding = nn.Sequential(
            nn.Linear(num_params, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        ) if num_params else None

    def forward(self, noisy_target, timestep, context, gaps, params=None):
        x = torch.cat([context, noisy_target], dim=1)
        cond = self.gap_embedding(gaps)
        if self.param_embedding is not None:
            cond = cond + self.param_embedding(params)
        return self.unet(x, timestep, class_labels=cond).sample


@torch.no_grad()
def sample_next(model, scheduler: DDIMScheduler, context, gaps, num_steps=50, eta=0.0, generator=None,
                params=None):
    """Draw next frames in normalized space given normalized context (B, K*C, H, W)."""
    C, H = model.config.num_channels, model.config.sample_size
    x = torch.randn((context.shape[0], C, H, H), generator=generator, device=context.device,
                    dtype=context.dtype)
    scheduler.set_timesteps(num_steps, device=context.device)
    for t in scheduler.timesteps:
        out = model(x, t, context, gaps, params)
        x = scheduler.step(out, t, x, eta=eta, generator=generator).prev_sample
    return x.clamp(-1, 1)
