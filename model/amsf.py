"""Adaptive Multi-Scale Fusion with CBAM for skip connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .patch_embed import circular_pad_w


class ChannelAttention(nn.Module):
    """Channel attention from CBAM."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        avg = x.mean(dim=(2, 3))  # (B, C)
        mx = x.amax(dim=(2, 3))   # (B, C)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention from CBAM."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        avg = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        mx = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        cat = torch.cat([avg, mx], dim=1)  # (B, 2, H, W)
        attn = torch.sigmoid(self.conv(cat))  # (B, 1, H, W)
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class AdaptiveMultiScaleFusion(nn.Module):
    """Three parallel conv branches (1x1, 3x3, 5x5) with adaptive softmax weights + CBAM.
    Used at skip connections in the decoder.
    Input/output: (B, H, W, C) format (Swin convention).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=0)  # will use circular pad
        self.conv5 = nn.Conv2d(dim, dim, 5, padding=0)  # will use circular pad

        # Softmax weight predictor: per-spatial-location weights for 3 branches
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, 3),
        )

        self.cbam = CBAM(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)  -- encoder skip features
        Returns:
            (B, H, W, C)
        """
        B, H, W, C = x.shape
        # Convert to (B, C, H, W) for convolutions
        x_conv = x.permute(0, 3, 1, 2).contiguous()

        # Branch 1: 1x1
        b1 = self.conv1(x_conv)

        # Branch 2: 3x3 with circular padding along width
        x_pad3 = circular_pad_w(x_conv, 1)
        x_pad3 = F.pad(x_pad3, (0, 0, 1, 1))  # height padding (zero)
        b2 = self.conv3(x_pad3)

        # Branch 3: 5x5 with circular padding along width
        x_pad5 = circular_pad_w(x_conv, 2)
        x_pad5 = F.pad(x_pad5, (0, 0, 2, 2))  # height padding (zero)
        b3 = self.conv5(x_pad5)

        # Adaptive weights via softmax
        weights = self.weight_net(x_conv)  # (B, 3)
        weights = F.softmax(weights, dim=-1)  # (B, 3)
        w1, w2, w3 = weights[:, 0], weights[:, 1], weights[:, 2]
        w1 = w1.view(B, 1, 1, 1)
        w2 = w2.view(B, 1, 1, 1)
        w3 = w3.view(B, 1, 1, 1)

        fused = w1 * b1 + w2 * b2 + w3 * b3  # (B, C, H, W)

        # CBAM
        fused = self.cbam(fused)

        # Back to (B, H, W, C)
        out = fused.permute(0, 2, 3, 1).contiguous()
        out = self.norm(out)
        return out
