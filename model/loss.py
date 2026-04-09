"""Loss functions for FLASH / FLASH+ super-resolution."""

import torch
import torch.nn.functional as F


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """L1 loss masked to valid pixels only.
    Args:
        pred: (B, 1, H, W) predicted range image
        target: (B, 1, H, W) ground truth range image
        mask: (B, 1, H, W) validity mask (1=valid, 0=invalid)
    Returns:
        scalar loss
    """
    loss = torch.abs(pred - target) * mask
    return loss.sum() / mask.sum().clamp(min=1)


# --------------- GAN losses (hinge) ---------------

def hinge_loss_disc(real_scores: torch.Tensor,
                    fake_scores: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator."""
    return (F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()) * 0.5


def hinge_loss_gen(fake_scores: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator (maximize fake scores)."""
    return -fake_scores.mean()


def distance_weighted_adv_loss(fake_scores: torch.Tensor,
                               range_image: torch.Tensor,
                               mask: torch.Tensor,
                               r_near: float = 30.0,
                               r_far: float = 60.0,
                               beta: float = 2.0) -> torch.Tensor:
    """Distance-weighted adversarial loss for generator.

    Weights long-range patches more heavily:
        w(r) = 1 + beta * clamp((r - r_near) / (r_far - r_near), 0, 1)

    Args:
        fake_scores: (B, 1, Hs, Ws) discriminator patch scores
        range_image: (B, 1, H, W) log-compressed range image
        mask: (B, 1, H, W) validity mask
        r_near, r_far: distance thresholds in meters
        beta: weighting factor
    """
    with torch.no_grad():
        # Convert to meters
        range_m = torch.expm1(range_image.clamp(min=0)) * mask  # (B, 1, H, W)
        # Distance weight
        t = ((range_m - r_near) / (r_far - r_near)).clamp(0, 1)
        w = 1.0 + beta * t  # (B, 1, H, W)
        # Downsample weight map to match score spatial resolution
        Hs, Ws = fake_scores.shape[2], fake_scores.shape[3]
        w = F.adaptive_avg_pool2d(w, (Hs, Ws))  # (B, 1, Hs, Ws)

    loss = -(fake_scores * w).mean()
    return loss


def freq_consistency_loss(weight_pairs: list) -> torch.Tensor:
    """Frequency consistency loss: encourage near/far filters to differ.

    L_freq = -sum(||W_near - W_far||_F) / num_pairs

    Args:
        weight_pairs: list of (W_near, W_far) weight tensors
    Returns:
        scalar loss (negative → minimizing encourages divergence)
    """
    if not weight_pairs:
        return torch.tensor(0.0)

    total = torch.tensor(0.0, device=weight_pairs[0][0].device)
    for w_near, w_far in weight_pairs:
        total = total - torch.norm(w_near - w_far, p="fro")
    return total / len(weight_pairs)
