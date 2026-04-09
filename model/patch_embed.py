"""Circular padding and row-based patch embedding for range images."""

import torch
import torch.nn as nn


def circular_pad_w(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Circular padding along width (last) dimension for 360-degree continuity.
    x: (B, C, H, W) or (B, H, W, C) -- works on last dim if specified."""
    if pad <= 0:
        return x
    left = x[..., -pad:]
    right = x[..., :pad]
    return torch.cat([left, x, right], dim=-1)


class PatchEmbed(nn.Module):
    """Row-based 1x4 patch embedding with circular padding."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 96,
                 patch_size: tuple = (1, 4)):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Circular padding needed along width for the conv
        self.pad_w = patch_size[1] // 2  # padding before conv not needed for stride=patch
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) e.g. (B, 1, 64, 1024)
        Returns:
            (B, H', W', embed_dim) e.g. (B, 64, 256, 96)
        """
        # Conv2d with stride=patch_size, no padding needed since W is divisible by 4
        x = self.proj(x)  # (B, embed_dim, H/ph, W/pw)
        x = x.permute(0, 2, 3, 1)  # (B, H', W', C)
        x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    """Downsample by merging 2x2 patches -> halve H, W, double C."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)
        Returns:
            (B, H/2, W/2, 2C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    """Upsample: double H, W, halve C."""

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 4 * (dim // 2), bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)
        Returns:
            (B, 2H, 2W, C/2)
        """
        B, H, W, C = x.shape
        x = self.expand(x)  # (B, H, W, 4*(C//2))
        new_c = C // 2
        x = x.view(B, H, W, 2, 2, new_c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, 2 * H, 2 * W, new_c)
        x = self.norm(x)
        return x
