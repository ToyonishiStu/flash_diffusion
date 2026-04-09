"""Swin Transformer block with non-square windows and circular shift."""

import torch
import torch.nn as nn
from typing import Optional
from .faa import FrequencyAwareAttention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: tuple) -> torch.Tensor:
    """Partition into non-overlapping windows.
    Args:
        x: (B, H, W, C)
        window_size: (wh, ww)
    Returns:
        (num_windows*B, wh, ww, C)
    """
    B, H, W, C = x.shape
    wh, ww = window_size
    x = x.view(B, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, wh, ww, C)
    return x


def window_reverse(windows: torch.Tensor, window_size: tuple,
                    H: int, W: int) -> torch.Tensor:
    """Reverse window partition.
    Args:
        windows: (num_windows*B, wh, ww, C)
        window_size: (wh, ww)
        H, W: original spatial dims
    Returns:
        (B, H, W, C)
    """
    wh, ww = window_size
    B = windows.shape[0] // (H // wh * W // ww)
    x = windows.view(B, H // wh, W // ww, wh, ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with FAA and non-square windows."""

    def __init__(self, dim: int, num_heads: int, window_size: tuple = (2, 8),
                 shift: bool = False, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, alpha_init: float = 0.1,
                 use_rafk: bool = False, rafk_mlp_hidden: int = 16):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift
        self.shift_size = (window_size[0] // 2, window_size[1] // 2) if shift else (0, 0)
        self.use_rafk = use_rafk

        self.norm1 = nn.LayerNorm(dim)
        self.attn = FrequencyAwareAttention(
            dim=dim, num_heads=num_heads, window_size=window_size,
            attn_drop=attn_drop, proj_drop=drop, alpha_init=alpha_init,
            use_rafk=use_rafk, rafk_mlp_hidden=rafk_mlp_hidden,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def _compute_attn_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        """Compute attention mask for shifted windows.
        Only height-boundary windows need masking (width wraps circularly).
        """
        if not self.shift:
            return None

        wh, ww = self.window_size
        sh, sw = self.shift_size

        # Create mask image
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -wh),
            slice(-wh, -sh),
            slice(-sh, None),
        )
        # Width is circular, so no masking needed along W
        # Only mask along height
        cnt = 0
        for h in h_slices:
            img_mask[:, h, :, :] = cnt
            cnt += 1

        # Partition to windows
        mask_windows = window_partition(img_mask, self.window_size)  # (nW, wh, ww, 1)
        mask_windows = mask_windows.view(-1, wh * ww)  # (nW, N)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        return attn_mask

    def _compute_window_feats(self, range_info, H, W):
        """Compute per-window 3D features for RAFK.
        Args:
            range_info: (range_map, valid_mask) each (B, H, W, 1)
        Returns:
            window_feats: (nW*B, 3) -- [v_w/H, r_bar_w/r_max, n_valid/n_total]
        """
        range_map, valid_mask = range_info  # (B, H, W, 1)
        B = range_map.shape[0]
        wh, ww = self.window_size
        nW_h, nW_w = H // wh, W // ww

        # Apply same cyclic shift to range_info
        if self.shift:
            sh, sw = self.shift_size
            range_map = torch.roll(range_map, shifts=(-sh, -sw), dims=(1, 2))
            valid_mask = torch.roll(valid_mask, shifts=(-sh, -sw), dims=(1, 2))

        # Window partition: (nW*B, wh, ww, 1)
        rm_win = window_partition(range_map, self.window_size)
        vm_win = window_partition(valid_mask, self.window_size)
        nW_total = rm_win.shape[0]  # nW*B

        # Feature 1: normalized row position v_w/H
        # Window index within each image
        nW = nW_h * nW_w
        # row_block for each window
        win_idx = torch.arange(nW, device=range_map.device)
        row_block = win_idx // nW_w  # (nW,)
        center_row = (row_block.float() * wh + wh / 2.0) / H  # (nW,)
        # Repeat for batch: (nW*B,)
        center_row = center_row.unsqueeze(0).expand(B, -1).reshape(-1)

        # Feature 2: mean range r_bar_w / r_max (80m)
        valid_sum = vm_win.view(nW_total, -1).sum(dim=1).clamp(min=1)  # (nW*B,)
        range_sum = (rm_win * vm_win).view(nW_total, -1).sum(dim=1)  # (nW*B,)
        mean_range = range_sum / valid_sum / 4.4  # log1p(80) ≈ 4.4, normalize

        # Feature 3: valid ratio n_valid / n_total
        n_total = wh * ww
        valid_ratio = vm_win.view(nW_total, -1).sum(dim=1) / n_total  # (nW*B,)

        # Stack: (nW*B, 3)
        window_feats = torch.stack([center_row, mean_range, valid_ratio], dim=1)
        return window_feats

    def forward(self, x: torch.Tensor, range_info=None) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)
            range_info: tuple (range_map, valid_mask) each (B, H, W, 1), or None
        Returns:
            (B, H, W, C)
        """
        B, H, W, C = x.shape
        wh, ww = self.window_size

        shortcut = x
        x = self.norm1(x)

        # Compute per-window features for RAFK
        window_feats = None
        if self.use_rafk and range_info is not None:
            window_feats = self._compute_window_feats(range_info, H, W)

        # Cyclic shift (circular along width)
        if self.shift:
            sh, sw = self.shift_size
            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))

        # Partition into windows
        x_windows = window_partition(x, self.window_size)  # (nW*B, wh, ww, C)
        x_windows = x_windows.view(-1, wh * ww, C)  # (nW*B, N, C)

        # Attention mask
        attn_mask = self._compute_attn_mask(H, W, x.device)

        # FAA attention
        attn_windows = self.attn(x_windows, mask=attn_mask,
                                 window_feats=window_feats)  # (nW*B, N, C)

        # Merge windows
        attn_windows = attn_windows.view(-1, wh, ww, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse shift
        if self.shift:
            x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))

        # Residual + MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, device=x.device, dtype=x.dtype)
        random_tensor = torch.floor(random_tensor + keep)
        return x / keep * random_tensor


class SwinStage(nn.Module):
    """A stage of Swin Transformer blocks (alternating regular/shifted windows)."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: tuple = (2, 8), mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, alpha_init: float = 0.1,
                 use_rafk: bool = False, rafk_mlp_hidden: int = 16):
        super().__init__()
        # Handle drop_path as list or scalar
        if isinstance(drop_path, (list, tuple)):
            dp_rates = drop_path
        else:
            dp_rates = [drop_path] * depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift=(i % 2 == 1), mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=dp_rates[i],
                alpha_init=alpha_init, use_rafk=use_rafk,
                rafk_mlp_hidden=rafk_mlp_hidden,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, range_info=None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, range_info=range_info)
        return x
