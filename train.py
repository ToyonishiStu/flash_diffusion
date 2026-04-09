"""Training loop for FLASH / FLASH+ LiDAR super-resolution."""

import os
import sys
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config.default import Config
from model.unet import FlashUNet
from model.loss import (
    masked_l1_loss, hinge_loss_disc, hinge_loss_gen,
    distance_weighted_adv_loss, freq_consistency_loss,
)
from data.dataset import create_dataloaders
from utils.misc import set_seed, get_device


def get_lr(epoch: int, config: Config) -> float:
    """Compute learning rate with linear warmup + cosine annealing with warm restarts."""
    if epoch < config.warmup_epochs:
        return config.lr * (epoch + 1) / config.warmup_epochs

    # Post-warmup: cosine annealing with warm restarts
    t = epoch - config.warmup_epochs
    T = config.restart_period
    cycle = t // T
    t_in_cycle = t % T

    # Peak LR decays by lr_decay_per_cycle each cycle
    peak_lr = config.lr * (config.lr_decay_per_cycle ** cycle)
    # Cosine annealing within the cycle
    lr = peak_lr * 0.5 * (1.0 + math.cos(math.pi * t_in_cycle / T))
    return max(lr, 1e-7)


def get_adv_weight(epoch: int, config: Config) -> float:
    """Linear warmup for adversarial loss weight."""
    if epoch < config.adv_warmup_epochs:
        return config.lambda_adv * epoch / max(config.adv_warmup_epochs, 1)
    return config.lambda_adv


def train_one_epoch(model, loader, optimizer, scaler, device, config,
                    disc=None, disc_optimizer=None, disc_scaler=None, epoch=0):
    model.train()
    if disc is not None:
        disc.train()

    total_l1 = 0.0
    total_adv = 0.0
    total_freq = 0.0
    total_disc = 0.0
    total_gen = 0.0
    count = 0

    adv_w = get_adv_weight(epoch, config) if config.use_mkdisc else 0.0

    for batch in loader:
        inp = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        # ---- Generator forward ----
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=config.mixed_precision):
            pred = model(inp)
            loss_l1 = masked_l1_loss(pred, target, mask)

            # Adversarial loss (generator side)
            loss_adv = torch.tensor(0.0, device=device)
            if disc is not None and adv_w > 0:
                fake_scores = disc(pred, mask)
                loss_adv = distance_weighted_adv_loss(
                    fake_scores, target, mask,
                    r_near=config.r_near, r_far=config.r_far,
                    beta=config.beta_dist_weight,
                )

            # Frequency consistency loss
            loss_freq = torch.tensor(0.0, device=device)
            if config.use_rafk:
                pairs = model.get_rafk_weight_pairs()
                loss_freq = freq_consistency_loss(pairs)

            loss_g = loss_l1 + adv_w * loss_adv + config.lambda_freq * loss_freq

        # Generator backward
        scaler.scale(loss_g).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # ---- Discriminator forward ----
        loss_d = torch.tensor(0.0, device=device)
        if disc is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=config.mixed_precision):
                real_scores = disc(target, mask)
                fake_scores = disc(pred.detach(), mask)
                loss_d = hinge_loss_disc(real_scores, fake_scores)

            disc_scaler.scale(loss_d).backward()
            disc_scaler.unscale_(disc_optimizer)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), config.disc_grad_clip)
            disc_scaler.step(disc_optimizer)
            disc_scaler.update()
            disc_optimizer.zero_grad(set_to_none=True)

        bs = inp.size(0)
        total_l1 += loss_l1.item() * bs
        total_adv += loss_adv.item() * bs
        total_freq += loss_freq.item() * bs
        total_disc += loss_d.item() * bs
        total_gen += loss_g.item() * bs
        count += bs

    n = max(count, 1)
    return {
        "l1": total_l1 / n,
        "adv": total_adv / n,
        "freq": total_freq / n,
        "disc": total_disc / n,
        "gen": total_gen / n,
    }


@torch.no_grad()
def validate(model, loader, device, config):
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in loader:
        inp = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=config.mixed_precision):
            pred = model(inp)
            loss = masked_l1_loss(pred, target, mask)

        total_loss += loss.item() * inp.size(0)
        count += inp.size(0)

    return total_loss / max(count, 1)


def save_checkpoint(model, optimizer, scaler, epoch, loss, path,
                    disc=None, disc_optimizer=None, disc_scaler=None):
    data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }
    if disc is not None:
        data["disc_state_dict"] = disc.state_dict()
    if disc_optimizer is not None:
        data["disc_optimizer_state_dict"] = disc_optimizer.state_dict()
    if disc_scaler is not None:
        data["disc_scaler_state_dict"] = disc_scaler.state_dict()
    torch.save(data, path)


def load_checkpoint(path, model, optimizer=None, scaler=None,
                    disc=None, disc_optimizer=None, disc_scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if disc is not None and "disc_state_dict" in ckpt:
        disc.load_state_dict(ckpt["disc_state_dict"])
    if disc_optimizer is not None and "disc_optimizer_state_dict" in ckpt:
        disc_optimizer.load_state_dict(ckpt["disc_optimizer_state_dict"])
    if disc_scaler is not None and "disc_scaler_state_dict" in ckpt:
        disc_scaler.load_state_dict(ckpt["disc_scaler_state_dict"])
    return ckpt.get("epoch", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Use dev config")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["baseline", "rafk", "mkdisc", "proposed"],
                        help="FLASH+ ablation variant")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()

    # Config
    if args.variant:
        config = Config.ablation(args.variant, dev=args.dev)
    else:
        config = Config.dev() if args.dev else Config()

    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        config.log_dir = args.log_dir

    set_seed(42)
    device = get_device()
    variant_str = args.variant or "flash"
    print(f"Device: {device}, Variant: {variant_str}, "
          f"batch_size={config.batch_size}, epochs={config.num_epochs}, "
          f"use_rafk={config.use_rafk}, use_mkdisc={config.use_mkdisc}")

    # Data
    train_loader, val_loader = create_dataloaders(config)

    # Generator
    model = FlashUNet(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Generator parameters: {params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr,
        betas=(0.9, 0.999), weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision)

    # Discriminator (MKDisc)
    disc = None
    disc_optimizer = None
    disc_scaler = None
    if config.use_mkdisc:
        from model.mkdisc import MetaKernelDiscriminator
        disc = MetaKernelDiscriminator(
            channels=config.mkdisc_channels,
            meta_kernel_size=config.mkdisc_meta_kernel_size,
            phi_hidden=config.mkdisc_phi_hidden,
            H=config.H, W=config.W,
            fov_up_deg=config.fov_up, fov_down_deg=-config.fov_down,  # fov_down is negative
            max_range=config.max_range,
        ).to(device)
        disc_params = sum(p.numel() for p in disc.parameters())
        print(f"Discriminator parameters: {disc_params:,}")

        disc_optimizer = torch.optim.AdamW(
            disc.parameters(), lr=config.disc_lr,
            betas=(0.0, 0.99), weight_decay=config.disc_weight_decay,
        )
        disc_scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision)

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(
            args.resume, model, optimizer, scaler,
            disc, disc_optimizer, disc_scaler,
        ) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Logging
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    writer = SummaryWriter(config.log_dir)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(start_epoch, config.num_epochs):
        t0 = time.time()

        # Set LR
        lr = get_lr(epoch, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Train
        losses = train_one_epoch(
            model, train_loader, optimizer, scaler, device, config,
            disc=disc, disc_optimizer=disc_optimizer, disc_scaler=disc_scaler,
            epoch=epoch,
        )

        # Validate
        val_loss = None
        if (epoch + 1) % config.val_interval == 0 or epoch == config.num_epochs - 1:
            val_loss = validate(model, val_loader, device, config)
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        log = (f"Epoch {epoch+1}/{config.num_epochs} | lr={lr:.2e} | "
               f"L1={losses['l1']:.5f} | G={losses['gen']:.5f}")
        if config.use_mkdisc:
            log += f" | Adv={losses['adv']:.5f} | D={losses['disc']:.5f}"
        if config.use_rafk:
            log += f" | Freq={losses['freq']:.5f}"
        if val_loss is not None:
            log += f" | val={val_loss:.5f}"
        log += f" | {elapsed:.1f}s"
        print(log)

        # TensorBoard
        writer.add_scalar("train/loss_l1", losses["l1"], epoch)
        writer.add_scalar("train/loss_gen", losses["gen"], epoch)
        writer.add_scalar("train/lr", lr, epoch)
        if config.use_mkdisc:
            writer.add_scalar("train/loss_adv", losses["adv"], epoch)
            writer.add_scalar("train/loss_disc", losses["disc"], epoch)
        if config.use_rafk:
            writer.add_scalar("train/loss_freq", losses["freq"], epoch)
        if val_loss is not None:
            writer.add_scalar("val/loss", val_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0 or epoch == config.num_epochs - 1:
            ckpt_path = os.path.join(config.checkpoint_dir, f"epoch_{epoch+1:04d}.pt")
            save_checkpoint(model, optimizer, scaler, epoch, losses["l1"], ckpt_path,
                            disc, disc_optimizer, disc_scaler)

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scaler, epoch, val_loss,
                            os.path.join(config.checkpoint_dir, "best.pt"),
                            disc, disc_optimizer, disc_scaler)

    writer.close()
    print(f"Training complete. Best val loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    main()
