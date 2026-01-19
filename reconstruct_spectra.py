#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import h5py
import numpy as np
import torch

from model_unet import UNet1D
from solver import linear_beta_schedule, ddpm_step


def parse_args():
    p = argparse.ArgumentParser(description="DPS reconstruction for Lya spectra.")
    p.add_argument("--h5", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--step-size", type=float, default=1e-2)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = UNet1D(in_ch=1, base_ch=64, depth=4, time_dim=256).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    betas = linear_beta_schedule(args.timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    with h5py.File(args.h5, "r") as h5:
        n_total = h5["flux_raw"].shape[0]
        n_use = min(args.n_samples, n_total)
        idxs = np.arange(n_use, dtype=np.int64)

        y = torch.from_numpy(h5["flux_noisy"][idxs]).float().to(device)
        sigma = torch.from_numpy(h5["sigma_map"][idxs]).float().to(device)
        scale = torch.from_numpy(h5["scale_factor"][idxs]).float().to(device)
        x_true = torch.from_numpy(h5["flux_raw"][idxs]).float()

    if scale.ndim == 1:
        scale = scale[:, None, None]

    x_t = torch.randn_like(y, device=device)

    for t_idx in range(args.timesteps - 1, -1, -1):
        t = torch.full((x_t.shape[0],), t_idx, device=device, dtype=torch.long)

        with torch.no_grad():
            eps = model(x_t, t.float())

        alpha_bar_t = alpha_bars[t].view(x_t.shape[0], 1, 1)
        x_t = x_t.detach()
        x_t.requires_grad_(True)
        x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

        resid = y - scale * x0_hat
        loss = torch.sum((resid ** 2) / (sigma ** 2))
        g = torch.autograd.grad(loss, x_t)[0]

        with torch.no_grad():
            x_prev = ddpm_step(x_t, eps, t, betas, alphas, alpha_bars)
            x_prev = x_prev - args.step_size * g
            x_t = x_prev

    flux_pred = (x_t * scale).detach().cpu().numpy()
    flux_noisy = y.detach().cpu().numpy()
    flux_true = x_true.numpy()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(args.out, "w") as h5:
        h5.create_dataset("flux_pred", data=flux_pred)
        h5.create_dataset("flux_true", data=flux_true)
        h5.create_dataset("flux_noisy", data=flux_noisy)

    print(f"[INFO] Wrote reconstructions: {args.out}")


if __name__ == "__main__":
    main()
