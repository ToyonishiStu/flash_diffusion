"""Meta-Kernel Discriminator (MKDisc) for FLASH+.

PatchGAN discriminator using Meta-Kernel convolutions that are aware of
the spherical geometry of LiDAR range images.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SphericalCoordBuilder(nn.Module):
    """Convert a range image (B,1,H,W) to spherical coordinates (B,3,H,W).

    Channels: [pitch, yaw, range] — all normalised to roughly [-1, 1] or [0, 1].
    """

    def __init__(self, H: int = 64, W: int = 1024,
                 fov_up_deg: float = 3.0, fov_down_deg: float = -25.0,
                 max_range: float = 80.0):
        super().__init__()
        fov_up = math.radians(fov_up_deg)
        fov_down = math.radians(fov_down_deg)
        fov_total = fov_up - fov_down

        # Pitch per row: top = fov_up, bottom = fov_down  (normalised to [0,1])
        pitch = torch.linspace(fov_up, fov_down, H) / fov_total  # (H,)
        # Yaw per col: [0, 2pi) normalised to [0,1]
        yaw = torch.linspace(0.0, 1.0, W + 1)[:W]  # (W,)

        # (1, 1, H, 1) and (1, 1, 1, W) — broadcastable to (B, 1, H, W)
        self.register_buffer("pitch_grid", pitch.view(1, 1, H, 1))
        self.register_buffer("yaw_grid", yaw.view(1, 1, 1, W))
        self.max_range_log = math.log1p(max_range)

    def forward(self, range_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            range_img: (B, 1, H, W) log-compressed range image
        Returns:
            coords: (B, 3, H, W)  [pitch, yaw, normalised_range]
        """
        B, _, H, W = range_img.shape
        pitch = self.pitch_grid.expand(B, 1, H, W)
        yaw = self.yaw_grid.expand(B, 1, H, W)
        norm_range = range_img / self.max_range_log  # [0, ~1]
        return torch.cat([pitch, yaw, norm_range], dim=1)


class MetaKernelLayer(nn.Module):
    """Meta-Kernel convolution layer.

    For each pixel i, aggregates features from its k×k neighbourhood j ∈ N(i)
    using geometry-aware weights: h'_i = W(∑_j Φ(γ(p_j, p_i)) ⊙ h_j)

    - γ: relative spherical coordinates (3D)
    - Φ: MLP (3 → hidden → in_channels) with Sigmoid
    - W: Linear + InstanceNorm + LeakyReLU
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, phi_hidden: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

        # Φ: relative coord → per-channel weight
        self.phi = nn.Sequential(
            nn.Linear(3, phi_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(phi_hidden, in_channels),
            nn.Sigmoid(),
        )

        # W: aggregate → output
        self.W = nn.Linear(in_channels * kernel_size * kernel_size, out_channels)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats:  (B, C_in, H, W)
            coords: (B, 3, H, W)
        Returns:
            out: (B, C_out, H, W)
        """
        B, C, H, W = feats.shape
        k = self.kernel_size

        # Circular padding along width, zero padding along height
        feats_pad = F.pad(feats, [self.pad, self.pad, 0, 0], mode="circular")
        feats_pad = F.pad(feats_pad, [0, 0, self.pad, self.pad], mode="constant", value=0)
        coords_pad = F.pad(coords, [self.pad, self.pad, 0, 0], mode="circular")
        coords_pad = F.pad(coords_pad, [0, 0, self.pad, self.pad], mode="constant", value=0)

        # Unfold: (B, C, H, W) -> (B, C*k*k, H*W)
        feats_unfold = F.unfold(feats_pad, kernel_size=k, stride=1)  # (B, C*k*k, H*W)
        feats_unfold = feats_unfold.view(B, C, k * k, H * W)  # (B, C, k*k, H*W)

        coords_unfold = F.unfold(coords_pad, kernel_size=k, stride=1)  # (B, 3*k*k, H*W)
        coords_unfold = coords_unfold.view(B, 3, k * k, H * W)  # (B, 3, k*k, H*W)

        # Centre pixel coords: use the middle element of the kernel
        mid = (k * k) // 2
        center_coords = coords_unfold[:, :, mid:mid+1, :]  # (B, 3, 1, H*W)

        # Relative coords γ
        rel_coords = coords_unfold - center_coords  # (B, 3, k*k, H*W)
        # (B, 3, k*k, H*W) -> (B*k*k*H*W, 3)
        rel_flat = rel_coords.permute(0, 2, 3, 1).reshape(-1, 3)

        # Φ(γ) -> (B*k*k*H*W, C)
        phi_out = self.phi(rel_flat)  # (B*k*k*H*W, C)
        phi_out = phi_out.view(B, k * k, H * W, C).permute(0, 3, 1, 2)  # (B, C, k*k, H*W)

        # Element-wise modulate and aggregate
        modulated = feats_unfold * phi_out  # (B, C, k*k, H*W)
        # Reshape for W: (B, C*k*k, H*W) -> (B, H*W, C*k*k)
        agg = modulated.view(B, C * k * k, H * W).permute(0, 2, 1)
        out = self.W(agg)  # (B, H*W, C_out)
        out = out.permute(0, 2, 1).view(B, self.out_channels, H, W)
        out = self.act(self.norm(out))
        return out


class MetaKernelDiscriminator(nn.Module):
    """PatchGAN discriminator with Meta-Kernel layers.

    Architecture: MKL(1→64) → AvgPool → MKL(64→128) → AvgPool →
                  MKL(128→256) → AvgPool → MKL(256→1)

    Input:  range image (B, 1, 64, 1024) + mask
    Output: patch scores (B, 1, 8, 128)
    """

    def __init__(self, channels=None, meta_kernel_size: int = 3,
                 phi_hidden: int = 32, H: int = 64, W: int = 1024,
                 fov_up_deg: float = 3.0, fov_down_deg: float = -25.0,
                 max_range: float = 80.0):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 1]

        self.coord_builder = SphericalCoordBuilder(
            H=H, W=W, fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg,
            max_range=max_range,
        )

        layers = []
        in_ch = 1  # range image has 1 channel
        for out_ch in channels:
            layers.append(MetaKernelLayer(in_ch, out_ch, meta_kernel_size, phi_hidden))
            in_ch = out_ch

        self.layers = nn.ModuleList(layers)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, range_img: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            range_img: (B, 1, H, W) log-compressed range
            mask: (B, 1, H, W) validity mask (optional, for masking input)
        Returns:
            scores: (B, 1, H/8, W/8) patch validity scores
        """
        if mask is not None:
            range_img = range_img * mask

        coords = self.coord_builder(range_img)  # (B, 3, H, W)
        x = range_img  # (B, 1, H, W)

        for i, layer in enumerate(self.layers):
            x = layer(x, coords)
            # Pool after every layer except the last
            if i < len(self.layers) - 1:
                x = self.pool(x)
                coords = self.pool(coords)

        return x
