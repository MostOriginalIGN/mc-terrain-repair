"""Deterministic U-Net for surface terrain repair."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TerrainRepairOutput:
    height_residual: torch.Tensor
    material_logits: torch.Tensor
    support: torch.Tensor


class RepairResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class RepairDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = RepairResBlock(in_channels, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.block(x)
        return h, self.downsample(h)


class RepairUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = RepairResBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class TerrainRepairUNet(nn.Module):
    """Residual deterministic repair model for height, material, and support proxy."""

    def __init__(self, num_material_classes: int, base_channels: int = 64):
        super().__init__()
        self.num_material_classes = num_material_classes
        scalar_channels = 1 + 1 + 1 + 1 + 2 + 1 + 1
        in_channels = scalar_channels + num_material_classes

        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = RepairDownBlock(base_channels, base_channels)
        self.down2 = RepairDownBlock(base_channels, base_channels * 2)
        self.mid = RepairResBlock(base_channels * 2, base_channels * 2)
        self.up1 = RepairUpBlock(base_channels * 2, base_channels * 2, base_channels)
        self.up2 = RepairUpBlock(base_channels, base_channels, base_channels)
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.height_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.material_head = nn.Conv2d(base_channels, num_material_classes, kernel_size=1)
        self.support_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(
        self,
        known_height: torch.Tensor,
        prefill_height: torch.Tensor,
        mask: torch.Tensor,
        known_material: torch.Tensor,
        known_support: torch.Tensor,
        boundary_distance: torch.Tensor,
        prefill_gradients: torch.Tensor,
        prefill_laplacian: torch.Tensor,
    ) -> TerrainRepairOutput:
        if known_material.ndim != 3:
            raise ValueError(f"known_material must have shape [B, H, W], got {tuple(known_material.shape)}")
        material_one_hot = F.one_hot(
            known_material.clamp(min=0, max=self.num_material_classes - 1).long(),
            num_classes=self.num_material_classes,
        ).permute(0, 3, 1, 2).float()

        x = torch.cat(
            [
                known_height,
                prefill_height,
                mask,
                boundary_distance,
                prefill_gradients,
                prefill_laplacian,
                known_support,
                material_one_hot,
            ],
            dim=1,
        )
        x = self.input_proj(x)
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        x = self.mid(x)
        x = self.up1(x, skip2)
        x = self.up2(x, skip1)
        x = F.silu(self.out_norm(x))

        return TerrainRepairOutput(
            height_residual=self.height_head(x),
            material_logits=self.material_head(x),
            support=torch.sigmoid(self.support_head(x)),
        )


__all__ = ["TerrainRepairOutput", "TerrainRepairUNet"]
