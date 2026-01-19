#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset_loader import LyaDataset, split_train_val
from model_unet import UNet1D


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, device="cpu"):
    return torch.linspace(beta_start, beta_end, timesteps, device=device)


def parse_args():
    p = argparse.ArgumentParser(description="Train 1D DDPM on Lya spectra (A100 AMP).")
    p.add_argument("--h5", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--ckpt-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    dataset = LyaDataset(args.h5)
    train_set, val_set = split_train_val(dataset, val_frac=args.val_frac, seed=args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = UNet1D(in_ch=1, base_ch=64, depth=4, time_dim=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    scaler = GradScaler()

    betas = linear_beta_schedule(args.timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    scratch = os.environ.get("SCRATCH", "/pscratch/sd/t/tanting")
    ckpt_dir = args.ckpt_dir or os.path.join(scratch, "experiments/checkpoints")
    log_dir = args.log_dir or os.path.join(scratch, "experiments/logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", ncols=100)
        train_loss = 0.0
        n_seen = 0

        for batch in pbar:
            x0 = batch["clean"].to(device)
            bsz = x0.shape[0]
            t = torch.randint(0, args.timesteps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            a_bar = alpha_bars[t].view(bsz, 1, 1)
            x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

            opt.zero_grad(set_to_none=True)
            with autocast():
                pred = model(x_t, t.float())
                loss = mse(pred, noise)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item() * bsz
            n_seen += bsz
            global_step += 1
            pbar.set_postfix(loss=train_loss / max(n_seen, 1))
            if global_step % 50 == 0:
                writer.add_scalar("train/loss", train_loss / max(n_seen, 1), global_step)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_n = 0
            for batch in val_loader:
                x0 = batch["clean"].to(device)
                bsz = x0.shape[0]
                t = torch.randint(0, args.timesteps, (bsz,), device=device, dtype=torch.long)
                noise = torch.randn_like(x0)
                a_bar = alpha_bars[t].view(bsz, 1, 1)
                x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
                with autocast():
                    pred = model(x_t, t.float())
                    loss = mse(pred, noise)
                val_loss += loss.item() * bsz
                val_n += bsz
            val_loss = val_loss / max(val_n, 1)
            writer.add_scalar("val/loss", val_loss, epoch)

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "train_loss": train_loss / max(n_seen, 1),
                "val_loss": val_loss,
                "timesteps": args.timesteps,
            },
            ckpt_path,
        )
        print(f"[INFO] Saved checkpoint: {ckpt_path} | val_loss={val_loss:.6f}")

    writer.close()


if __name__ == "__main__":
    main()
