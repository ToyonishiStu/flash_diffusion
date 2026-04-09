"""PyTorch Dataset for range image super-resolution."""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config.default import Config


class RangeImageDataset(Dataset):
    """Dataset that loads preprocessed range image .npy files."""

    def __init__(self, file_list: list, config: Config = None):
        """
        Args:
            file_list: list of paths to *_range.npy files
            config: Config object
        """
        self.file_list = sorted(file_list)
        self.config = config or Config()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        range_path = self.file_list[idx]
        mask_path = range_path.replace("_range.npy", "_mask.npy")

        range_img = np.load(range_path)   # (64, 1024)
        mask = np.load(mask_path)          # (64, 1024)

        # Target: full 64-row range image
        target = torch.from_numpy(range_img).unsqueeze(0)  # (1, 64, 1024)
        mask_t = torch.from_numpy(mask).unsqueeze(0)        # (1, 64, 1024)

        # Input: subsample to 16 rows, then bilinear upsample back to 64
        sr = self.config.sr_factor
        lr_img = range_img[::sr, :]  # (16, 1024)
        lr_t = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0)  # (1, 1, 16, 1024)
        input_t = F.interpolate(lr_t, size=(self.config.H, self.config.W),
                                mode="bilinear", align_corners=False)
        input_t = input_t.squeeze(0)  # (1, 64, 1024)

        return {
            "input": input_t,
            "target": target,
            "mask": mask_t,
            "path": range_path,
        }


def gather_files(processed_root: str, drive_names: list = None) -> list:
    """Gather all *_range.npy files, optionally filtered by drive names."""
    all_files = sorted(glob.glob(
        os.path.join(processed_root, "**", "*_range.npy"), recursive=True
    ))
    if drive_names is not None:
        filtered = []
        for f in all_files:
            for d in drive_names:
                if d in f:
                    filtered.append(f)
                    break
        return filtered
    return all_files


def load_drive_list(path: str) -> list:
    """Load drive names from a text file (one per line)."""
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def create_dataloaders(config: Config) -> tuple:
    """Create train and val DataLoaders.

    Strategy:
    - If train_drives_file / test_drives_file exist: split by drive name
    - Otherwise (dev mode): split single drive 80/20 by frame index
    """
    train_drives = load_drive_list(config.train_drives_file)
    test_drives = load_drive_list(config.test_drives_file)

    if train_drives is not None and test_drives is not None:
        train_files = gather_files(config.processed_root, train_drives)
        val_files = gather_files(config.processed_root, test_drives)
    else:
        # Dev mode: split all files 80/20
        all_files = gather_files(config.processed_root)
        split = int(len(all_files) * 0.8)
        train_files = all_files[:split]
        val_files = all_files[split:]

    print(f"Train: {len(train_files)} frames, Val: {len(val_files)} frames")

    train_ds = RangeImageDataset(train_files, config)
    val_ds = RangeImageDataset(val_files, config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
