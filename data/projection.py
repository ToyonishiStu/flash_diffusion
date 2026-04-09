"""Spherical projection: KITTI Velodyne .bin -> range image."""

import numpy as np
from config.default import Config


def load_velodyne_bin(path: str) -> np.ndarray:
    """Load KITTI velodyne .bin file. Returns (N, 4) array [x, y, z, reflectance]."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts


def spherical_projection(
    points: np.ndarray,
    H: int = 64,
    W: int = 1024,
    fov_up_rad: float = None,
    fov_down_rad: float = None,
    fov_total_rad: float = None,
    max_range: float = 80.0,
    config: Config = None,
) -> tuple:
    """
    Project 3D points to a 2D range image via spherical projection.

    Args:
        points: (N, 4) array [x, y, z, reflectance]
        H, W: range image dimensions
        config: if provided, overrides H, W, fov params

    Returns:
        range_img: (H, W) float32, log-compressed range values
        mask: (H, W) float32, 1 where valid, 0 where empty
    """
    if config is not None:
        H = config.H
        W = config.W
        fov_up_rad = config.fov_up_rad
        fov_down_rad = config.fov_down_rad
        fov_total_rad = config.fov_total_rad
        max_range = config.max_range
    else:
        if fov_up_rad is None:
            cfg = Config()
            fov_up_rad = cfg.fov_up_rad
            fov_down_rad = cfg.fov_down_rad
            fov_total_rad = cfg.fov_total_rad

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Range
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Filter invalid points
    valid = (r > 0.1) & (r < max_range)
    x, y, z, r = x[valid], y[valid], z[valid], r[valid]

    # Pitch (elevation) and yaw (azimuth)
    pitch = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    yaw = np.arctan2(y, x)

    # Row index: top = fov_up, bottom = fov_down
    row = ((fov_up_rad - pitch) / fov_total_rad * H).astype(np.int32)
    # Column index: full 360 degrees
    col = ((np.pi - yaw) / (2.0 * np.pi) * W).astype(np.int32)

    # Clamp
    row = np.clip(row, 0, H - 1)
    col = np.clip(col, 0, W - 1)

    # Sort by descending range so closer points overwrite farther ones
    order = np.argsort(-r)
    row = row[order]
    col = col[order]
    r = r[order]

    # Fill range image
    range_img = np.zeros((H, W), dtype=np.float32)
    range_img[row, col] = r

    # Log compression
    mask = (range_img > 0).astype(np.float32)
    range_img = np.log1p(range_img) * mask  # log(r + 1), zero stays zero

    return range_img, mask


def bin_to_range_image(bin_path: str, config: Config = None) -> tuple:
    """Convenience: load .bin and project to range image."""
    points = load_velodyne_bin(bin_path)
    return spherical_projection(points, config=config)
