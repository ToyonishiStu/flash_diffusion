from dataclasses import dataclass, field
from typing import Tuple, Optional
import math


@dataclass
class Config:
    # Data
    data_root: str = "kitti_raw"
    processed_root: str = "kitti_processed"
    H: int = 64
    W: int = 1024
    fov_up: float = 3.0      # degrees
    fov_down: float = -25.0   # degrees
    sr_factor: int = 4        # 16 -> 64 rows
    max_range: float = 80.0   # meters

    # Model
    in_channels: int = 1
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 6, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    window_size: Tuple[int, int] = (2, 8)
    patch_size: Tuple[int, int] = (1, 4)
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    faa_alpha_init: float = 0.1

    # Training
    batch_size: int = 8
    num_epochs: int = 600
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 60
    restart_period: int = 600
    lr_decay_per_cycle: float = 0.7
    grad_clip: float = 1.0

    # Device adaptations
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    num_workers: int = 4

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    train_drives_file: Optional[str] = None
    test_drives_file: Optional[str] = None

    # FLASH+ mode
    use_rafk: bool = False
    use_mkdisc: bool = False

    # RAFK
    rafk_mlp_hidden: int = 16

    # MKDisc
    mkdisc_channels: list = field(default_factory=lambda: [64, 128, 256, 1])
    mkdisc_meta_kernel_size: int = 3
    mkdisc_phi_hidden: int = 32

    # Loss weights
    lambda_adv: float = 0.1
    lambda_freq: float = 0.01
    beta_dist_weight: float = 2.0
    r_near: float = 30.0
    r_far: float = 60.0

    # Disc training
    disc_lr: float = 2e-4
    disc_weight_decay: float = 0.01
    disc_grad_clip: float = 1.0
    adv_warmup_epochs: int = 10

    # Eval
    val_interval: int = 5
    save_interval: int = 10
    voxel_size: float = 0.1
    cd_threshold: float = 0.1

    @property
    def fov_up_rad(self) -> float:
        return self.fov_up * math.pi / 180.0

    @property
    def fov_down_rad(self) -> float:
        return self.fov_down * math.pi / 180.0

    @property
    def fov_total_rad(self) -> float:
        return self.fov_up_rad - self.fov_down_rad

    @property
    def lr_rows(self) -> int:
        return self.H // self.sr_factor

    @classmethod
    def dev(cls) -> "Config":
        return cls(
            batch_size=1,
            gradient_checkpointing=True,
            num_workers=2,
            num_epochs=10,
            val_interval=1,
            save_interval=5,
        )

    @classmethod
    def ablation(cls, variant: str, dev: bool = False) -> "Config":
        """Create config for ablation variant.

        Args:
            variant: One of 'baseline', 'rafk', 'mkdisc', 'proposed'.
            dev: If True, use dev-mode settings (small batch, few epochs).
        """
        flags = {
            "baseline": dict(use_rafk=False, use_mkdisc=False),
            "rafk":     dict(use_rafk=True,  use_mkdisc=False),
            "mkdisc":   dict(use_rafk=False, use_mkdisc=True),
            "proposed":  dict(use_rafk=True,  use_mkdisc=True),
        }
        if variant not in flags:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(flags.keys())}")

        if dev:
            cfg = cls.dev()
        else:
            cfg = cls()

        for k, v in flags[variant].items():
            setattr(cfg, k, v)
        return cfg
