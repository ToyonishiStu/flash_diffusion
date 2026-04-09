"""FLASH U-Net: Swin Transformer encoder-decoder with FAA and AMSF."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from .patch_embed import PatchEmbed, PatchMerge, PatchExpand
from .swin_block import SwinStage
from .amsf import AdaptiveMultiScaleFusion
from config.default import Config


class FlashUNet(nn.Module):
    """Full FLASH super-resolution model."""

    def __init__(self, config: Config = None):
        super().__init__()
        if config is None:
            config = Config()

        self.config = config
        C = config.embed_dim
        depths = config.depths
        num_heads = config.num_heads
        window_size = config.window_size

        # Stochastic depth rates
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, total_depth)]
        dpr_splits = []
        idx = 0
        for d in depths:
            dpr_splits.append(dpr[idx:idx + d])
            idx += d

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=config.in_channels,
            embed_dim=C,
            patch_size=config.patch_size,
        )

        self.use_rafk = config.use_rafk

        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dims = [C * (2 ** i) for i in range(len(depths))]  # [96, 192, 384, 768]

        for i in range(len(depths)):
            stage = SwinStage(
                dim=dims[i], depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=config.mlp_ratio,
                drop=config.drop_rate, attn_drop=config.attn_drop_rate,
                drop_path=dpr_splits[i], alpha_init=config.faa_alpha_init,
                use_rafk=config.use_rafk, rafk_mlp_hidden=config.rafk_mlp_hidden,
            )
            self.encoder_stages.append(stage)
            if i < len(depths) - 1:
                self.downsamples.append(PatchMerge(dims[i]))

        # Decoder stages (reverse order, skip bottleneck)
        self.decoder_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.amsf_modules = nn.ModuleList()
        self.skip_projections = nn.ModuleList()

        for i in range(len(depths) - 2, -1, -1):
            self.upsamples.append(PatchExpand(dims[i + 1]))
            self.amsf_modules.append(AdaptiveMultiScaleFusion(dims[i]))
            self.skip_projections.append(nn.Linear(2 * dims[i], dims[i]))
            stage = SwinStage(
                dim=dims[i], depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=config.mlp_ratio,
                drop=config.drop_rate, attn_drop=config.attn_drop_rate,
                drop_path=dpr_splits[i], alpha_init=config.faa_alpha_init,
                use_rafk=config.use_rafk, rafk_mlp_hidden=config.rafk_mlp_hidden,
            )
            self.decoder_stages.append(stage)

        # Final expansion: reverse patch embedding (1x4)
        pw = config.patch_size[1]
        self.final_expand = nn.Linear(C, pw * C, bias=False)
        self.final_norm = nn.LayerNorm(C)
        self.head = nn.Linear(C, config.in_channels)

        self.use_checkpoint = config.gradient_checkpointing
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _forward_stage(self, stage, x, range_info=None):
        if self.use_checkpoint and self.training:
            # gradient checkpointing doesn't support kwargs well, pass None if no range_info
            return checkpoint.checkpoint(stage, x, range_info, use_reentrant=False)
        return stage(x, range_info=range_info)

    def _build_range_info(self, x_input: torch.Tensor):
        """Build range_info from raw input for RAFK.
        Args:
            x_input: (B, 1, H, W) raw log-compressed range image
        Returns:
            range_map: (B, H', W', 1) at patch-embedded resolution
            valid_mask: (B, H', W', 1) at patch-embedded resolution
        """
        # Undo log compression to get meters, detached
        with torch.no_grad():
            valid_mask = (x_input > 0).float()  # (B, 1, H, W)
            range_map = x_input.clamp(min=0).detach()  # keep log-compressed for normalization

            # Pool to patch-embedded resolution
            ph, pw = self.config.patch_size
            range_map = F.avg_pool2d(range_map, kernel_size=(ph, pw), stride=(ph, pw))
            valid_mask = F.avg_pool2d(valid_mask, kernel_size=(ph, pw), stride=(ph, pw))

            # (B, 1, H', W') -> (B, H', W', 1)
            range_map = range_map.permute(0, 2, 3, 1)
            valid_mask = valid_mask.permute(0, 2, 3, 1)

        return range_map, valid_mask

    @staticmethod
    def _pool_range_info(range_info):
        """Downsample range_info by 2x2 for next encoder level."""
        range_map, valid_mask = range_info
        # (B, H, W, 1) -> (B, 1, H, W) for pooling
        rm = range_map.permute(0, 3, 1, 2)
        vm = valid_mask.permute(0, 3, 1, 2)
        rm = F.avg_pool2d(rm, kernel_size=2, stride=2)
        vm = F.avg_pool2d(vm, kernel_size=2, stride=2)
        return rm.permute(0, 2, 3, 1), vm.permute(0, 2, 3, 1)

    def get_rafk_weight_pairs(self):
        """Collect all RAFK (W_near, W_far) pairs from all stages."""
        pairs = []
        for stage in self.encoder_stages:
            for blk in stage.blocks:
                pairs.extend(blk.attn.get_rafk_weight_pairs())
        for stage in self.decoder_stages:
            for blk in stage.blocks:
                pairs.extend(blk.attn.get_rafk_weight_pairs())
        return pairs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) e.g. (B, 1, 64, 1024)
        Returns:
            (B, 1, H, W)
        """
        # Build range_info for RAFK
        range_info = None
        encoder_range_infos = []
        if self.use_rafk:
            range_info = self._build_range_info(x)

        # Patch embedding: (B, 1, 64, 1024) -> (B, 64, 256, 96)
        x = self.patch_embed(x)

        # Encoder
        skips = []
        for i, stage in enumerate(self.encoder_stages):
            x = self._forward_stage(stage, x, range_info)
            if i < len(self.encoder_stages) - 1:
                skips.append(x)
                if range_info is not None:
                    encoder_range_infos.append(range_info)
                    range_info = self._pool_range_info(range_info)
                x = self.downsamples[i](x)

        # Decoder
        for i, (up, amsf, proj, stage) in enumerate(zip(
            self.upsamples, self.amsf_modules,
            self.skip_projections, self.decoder_stages
        )):
            x = up(x)
            skip = skips[-(i + 1)]
            skip = amsf(skip)
            x = torch.cat([x, skip], dim=-1)
            x = proj(x)
            # Reuse encoder range_info at matching resolution
            dec_range_info = encoder_range_infos[-(i + 1)] if encoder_range_infos else None
            x = self._forward_stage(stage, x, dec_range_info)

        # Final expansion: (B, 64, 256, 96) -> (B, 64, 1024, 96) -> (B, 64, 1024, 1)
        B, H, W, C = x.shape
        pw = self.config.patch_size[1]
        x = self.final_expand(x)  # (B, H, W, pw*C)
        x = x.view(B, H, W * pw, C)  # (B, 64, 1024, 96)
        x = self.final_norm(x)
        x = self.head(x)  # (B, 64, 1024, 1)
        x = x.permute(0, 3, 1, 2)  # (B, 1, 64, 1024)
        return x
