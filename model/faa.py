"""Frequency-Aware Attention: spatial MSA + FFT dual-branch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAwareAttention(nn.Module):
    """Dual-branch attention: spatial windowed MSA + FFT branch.
    Operates on tokens within a single window.
    """

    def __init__(self, dim: int, num_heads: int, window_size: tuple,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 alpha_init: float = 0.1, use_rafk: bool = False,
                 rafk_mlp_hidden: int = 16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_h, self.window_w = window_size
        self.use_rafk = use_rafk

        # Spatial branch
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_h - 1) * (2 * self.window_w - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(self.window_h)
        coords_w = torch.arange(self.window_w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, wh, ww)
        coords_flat = coords.view(2, -1)  # (2, wh*ww)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_h - 1
        relative_coords[:, :, 1] += self.window_w - 1
        relative_coords[:, :, 0] *= 2 * self.window_w - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        # FFT branch (FLASH paper Eq.(7): Conv2d on amplitude → sigmoid mask)
        if use_rafk:
            self.conv_near = nn.Conv2d(1, 1, kernel_size=3, padding=1)
            self.conv_far = nn.Conv2d(1, 1, kernel_size=3, padding=1)
            self.rafk_mlp = nn.Sequential(
                nn.Linear(3, rafk_mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(rafk_mlp_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.freq_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        # Learnable fusion weight
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def spatial_attention(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Standard windowed multi-head self-attention.
        Args:
            x: (num_windows*B, N, C) where N = wh*ww
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1)  # (heads, N, N)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return out

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def fft_branch(self, x: torch.Tensor, window_feats: torch.Tensor = None) -> torch.Tensor:
        """FFT branch per FLASH paper Eqs. (6)-(9).
        Args:
            x: (num_windows*B, N, C) where N = wh*ww
            window_feats: (num_windows*B, 3) per-window range features (RAFK only)
        Note: Decorated with custom_fwd to force float32, avoiding ComplexHalf issues.
        """
        B_, N, C = x.shape
        wh, ww = self.window_h, self.window_w

        # Eq.(6): channel-mean → 2D FFT (1 channel)
        x_mean = x.view(B_, wh, ww, C).mean(dim=-1)        # (B_, wh, ww)
        x_fft = torch.fft.rfft2(x_mean, norm="ortho")       # (B_, wh, ww//2+1) complex

        # Eq.(7): amplitude → Conv2d → sigmoid (multiplicative mask)
        amp = x_fft.abs().unsqueeze(1)                      # (B_, 1, wh, ww//2+1)

        if self.use_rafk and window_feats is not None:
            F_near = torch.sigmoid(self.conv_near(amp)).squeeze(1)   # (B_, wh, ww//2+1)
            F_far = torch.sigmoid(self.conv_far(amp)).squeeze(1)     # (B_, wh, ww//2+1)
            alpha_b = self.rafk_mlp(window_feats).view(B_, 1, 1)     # (B_, 1, 1)
            # α=0 → conv_near dominant (near range), α=1 → conv_far dominant (far range)
            F_attn = (1.0 - alpha_b) * F_near + alpha_b * F_far
        else:
            F_attn = torch.sigmoid(self.freq_conv(amp)).squeeze(1)   # (B_, wh, ww//2+1)

        # Eq.(8): freq coefficients × attention mask → inverse FFT
        x_fft_out = x_fft * F_attn                                   # complex × real
        x_spatial = torch.fft.irfft2(x_fft_out, s=(wh, ww), norm="ortho")  # (B_, wh, ww)

        # Eq.(9): 1ch → C ch broadcast (view-only expand; gradients summed across C)
        x_out = x_spatial.unsqueeze(-1).expand(-1, -1, -1, C)        # (B_, wh, ww, C)
        return x_out.reshape(B_, N, C)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                window_feats: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (num_windows*B, N, C)
            mask: attention mask for shifted windows
            window_feats: (num_windows*B, 3) per-window range features (RAFK only)
        Returns:
            (num_windows*B, N, C)
        """
        spatial = self.spatial_attention(x, mask)
        freq = self.fft_branch(x, window_feats=window_feats)

        alpha = torch.sigmoid(self.alpha)
        out = alpha * freq + (1.0 - alpha) * spatial
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def get_rafk_weight_pairs(self):
        """Return list of (W_near, W_far) Conv2d weight pairs. Shape: (1, 1, 3, 3)."""
        if self.use_rafk:
            return [(self.conv_near.weight, self.conv_far.weight)]
        return []
