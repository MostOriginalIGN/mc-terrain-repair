"""2D U-Net for masked terrain diffusion."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TerrainDiffusionOutput:
    noise_pred: torch.Tensor
    material_logits: torch.Tensor


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1))
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.block = ResBlock(in_channels, out_channels, time_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.block(x, time_emb)
        return h, self.downsample(h)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = ResBlock(out_channels + skip_channels, out_channels, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, time_emb)


class TerrainDiffusionUNet(nn.Module):
    def __init__(self, num_material_classes: int, base_channels: int = 64, gradient_channels: int = 2):
        super().__init__()
        self.num_material_classes = num_material_classes
        in_channels = 1 + 1 + 1 + gradient_channels + num_material_classes
        time_dim = base_channels * 4

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = DownBlock(base_channels, base_channels, time_dim)
        self.down2 = DownBlock(base_channels, base_channels * 2, time_dim)
        self.mid = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_dim)
        self.up2 = UpBlock(base_channels, base_channels, base_channels, time_dim)
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.noise_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.material_head = nn.Conv2d(base_channels, num_material_classes, kernel_size=1)

    def forward(
        self,
        noisy_height: torch.Tensor,
        known_height: torch.Tensor,
        mask: torch.Tensor,
        known_material: torch.Tensor,
        gradients: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> TerrainDiffusionOutput:
        if known_material.ndim != 3:
            raise ValueError(f"known_material must have shape [B, H, W], got {tuple(known_material.shape)}")

        material_one_hot = F.one_hot(
            known_material.clamp(min=0, max=self.num_material_classes - 1).long(),
            num_classes=self.num_material_classes,
        ).permute(0, 3, 1, 2).float()

        x = torch.cat([noisy_height, known_height, mask, gradients, material_one_hot], dim=1)
        time_emb = self.time_embed(timesteps)

        x = self.input_proj(x)
        skip1, x = self.down1(x, time_emb)
        skip2, x = self.down2(x, time_emb)
        x = self.mid(x, time_emb)
        x = self.up1(x, skip2, time_emb)
        x = self.up2(x, skip1, time_emb)
        x = F.silu(self.out_norm(x))

        return TerrainDiffusionOutput(
            noise_pred=self.noise_head(x),
            material_logits=self.material_head(x),
        )
