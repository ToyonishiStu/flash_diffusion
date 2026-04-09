"""Evaluation metrics: MAE, Chamfer Distance, IoU, Precision, Recall, F1."""

import numpy as np
from scipy.spatial import KDTree


def compute_mae(pred_range: np.ndarray, gt_range: np.ndarray,
                mask: np.ndarray) -> float:
    """Mean Absolute Error in meters (after undoing log compression).
    Args:
        pred_range: (H, W) log-compressed predicted range image
        gt_range: (H, W) log-compressed ground truth range image
        mask: (H, W) validity mask
    """
    valid = mask > 0
    if valid.sum() == 0:
        return 0.0
    pred_m = np.expm1(pred_range[valid])  # undo log(r+1)
    gt_m = np.expm1(gt_range[valid])
    return float(np.abs(pred_m - gt_m).mean())


def compute_chamfer_distance(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """Chamfer Distance between two point clouds.
    Args:
        pred_pts: (N, 3)
        gt_pts: (M, 3)
    Returns:
        CD = mean(pred->gt) + mean(gt->pred)
    """
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    tree_gt = KDTree(gt_pts)
    tree_pred = KDTree(pred_pts)

    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)

    cd = float(np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2))
    return cd


def compute_iou(pred_pts: np.ndarray, gt_pts: np.ndarray,
                voxel_size: float = 0.1) -> float:
    """IoU of voxelized point clouds.
    Args:
        pred_pts: (N, 3)
        gt_pts: (M, 3)
        voxel_size: voxel resolution in meters
    """
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0

    pred_voxels = set(map(tuple, np.floor(pred_pts / voxel_size).astype(np.int64)))
    gt_voxels = set(map(tuple, np.floor(gt_pts / voxel_size).astype(np.int64)))

    intersection = len(pred_voxels & gt_voxels)
    union = len(pred_voxels | gt_voxels)
    return intersection / max(union, 1)


def compute_precision_recall_f1(pred_pts: np.ndarray, gt_pts: np.ndarray,
                                threshold: float = 0.1) -> tuple:
    """Precision, Recall, F1 at a distance threshold.
    Args:
        pred_pts: (N, 3)
        gt_pts: (M, 3)
        threshold: distance threshold in meters
    Returns:
        (precision, recall, f1)
    """
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0, 0.0, 0.0

    tree_gt = KDTree(gt_pts)
    tree_pred = KDTree(pred_pts)

    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)

    precision = float(np.mean(d_pred_to_gt < threshold))
    recall = float(np.mean(d_gt_to_pred < threshold))
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def compute_all_metrics(pred_range: np.ndarray, gt_range: np.ndarray,
                        mask: np.ndarray,
                        pred_pts: np.ndarray, gt_pts: np.ndarray,
                        voxel_size: float = 0.1,
                        threshold: float = 0.1) -> dict:
    """Compute all metrics at once."""
    mae = compute_mae(pred_range, gt_range, mask)
    cd = compute_chamfer_distance(pred_pts, gt_pts)
    iou = compute_iou(pred_pts, gt_pts, voxel_size)
    prec, rec, f1 = compute_precision_recall_f1(pred_pts, gt_pts, threshold)
    return {
        "mae": mae,
        "chamfer_distance": cd,
        "iou": iou,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def compute_mae_by_distance(pred_range: np.ndarray, gt_range: np.ndarray,
                            mask: np.ndarray, config=None,
                            distance_ranges: list = None) -> dict:
    """Compute MAE split by distance range.
    Args:
        pred_range, gt_range: (H, W) log-compressed range images
        mask: (H, W) validity mask
        config: Config object (for FOV params); uses defaults if None
        distance_ranges: list of (min_m, max_m) tuples
    Returns:
        dict of {range_label: float mae_in_meters}
    """
    if distance_ranges is None:
        distance_ranges = [(0, 30), (30, 60)]

    valid = mask > 0
    gt_m = np.expm1(gt_range)  # meters
    pred_m = np.expm1(pred_range)

    results = {}
    for dmin, dmax in distance_ranges:
        label = f"{dmin}-{dmax}m"
        sel = valid & (gt_m >= dmin) & (gt_m < dmax)
        if sel.sum() > 0:
            results[label] = float(np.abs(pred_m[sel] - gt_m[sel]).mean())
        else:
            results[label] = float("nan")
    return results


def compute_metrics_by_distance(pred_pts: np.ndarray, gt_pts: np.ndarray,
                                 distance_ranges: list = None) -> dict:
    """Compute metrics split by distance from origin.
    Args:
        distance_ranges: list of (min_dist, max_dist) tuples
    Returns:
        dict of {range_label: {metric: value}}
    """
    if distance_ranges is None:
        distance_ranges = [(0, 10), (10, 30), (30, 50), (50, 80)]

    results = {}
    for dmin, dmax in distance_ranges:
        label = f"{dmin}-{dmax}m"
        pred_r = np.linalg.norm(pred_pts, axis=1)
        gt_r = np.linalg.norm(gt_pts, axis=1)
        pred_sub = pred_pts[(pred_r >= dmin) & (pred_r < dmax)]
        gt_sub = gt_pts[(gt_r >= dmin) & (gt_r < dmax)]
        if len(pred_sub) > 0 and len(gt_sub) > 0:
            cd = compute_chamfer_distance(pred_sub, gt_sub)
            iou = compute_iou(pred_sub, gt_sub)
            prec, rec, f1 = compute_precision_recall_f1(pred_sub, gt_sub)
            results[label] = {"cd": cd, "iou": iou, "precision": prec,
                              "recall": rec, "f1": f1,
                              "num_pred": len(pred_sub), "num_gt": len(gt_sub)}
        else:
            results[label] = {"cd": float("inf"), "iou": 0, "precision": 0,
                              "recall": 0, "f1": 0,
                              "num_pred": len(pred_sub), "num_gt": len(gt_sub)}
    return results
