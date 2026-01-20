#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_unet import UNet1D
from solver import ddpm_step, linear_beta_schedule


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate generated spectra against real test spectra."
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    p.add_argument("--h5", type=str, required=True, help="Path to HDF5 test dataset.")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=str, default="validation_report.pdf")
    return p.parse_args()


def load_real_samples(h5_path, n_samples, seed):
    with h5py.File(h5_path, "r") as h5:
        n_total = int(h5["flux_raw"].shape[0])
        n_use = min(n_samples, n_total)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=n_use, replace=False))
        flux_raw = np.asarray(h5["flux_raw"][idx], dtype=np.float32)
        scale = np.asarray(h5["scale_factor"][idx], dtype=np.float32)
    flux_norm = flux_raw / scale[:, None, None]
    return flux_norm


def ddpm_sample(model, n_samples, n_pix, timesteps, batch_size, device):
    betas = linear_beta_schedule(timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    samples = []
    for start in range(0, n_samples, batch_size):
        bsz = min(batch_size, n_samples - start)
        x_t = torch.randn((bsz, 1, n_pix), device=device)
        for t_idx in tqdm(
            range(timesteps - 1, -1, -1),
            desc=f"Sampling {start + bsz}/{n_samples}",
            ncols=100,
            leave=False,
        ):
            t = torch.full((bsz,), t_idx, device=device, dtype=torch.long)
            with torch.no_grad():
                eps = model(x_t, t.float())
                x_t = ddpm_step(x_t, eps, t, betas, alphas, alpha_bars)
        samples.append(x_t.detach().cpu().numpy())
    return np.concatenate(samples, axis=0)


def compute_power_spectrum(flux):
    fft = np.fft.rfft(flux, axis=-1)
    power = np.abs(fft) ** 2
    mean_power = power.mean(axis=0)
    k = np.fft.rfftfreq(flux.shape[-1], d=1.0)
    return k, mean_power


def curvature_stat(flux, eps=1e-6):
    log_flux = np.log(np.clip(flux, eps, None))
    d1 = np.gradient(log_flux, axis=-1)
    d2 = np.gradient(d1, axis=-1)
    return d2


def lower_bound_fraction(curvature, n_pix, r_out=241.0, l_min=2000.0, l_max=8000.0):
    wave = np.linspace(l_min, l_max, n_pix)
    loglam = np.log(wave)
    dloglam = np.mean(np.diff(loglam))
    sigma_loglam = 1.0 / (2.355 * r_out)
    sigma_pix = sigma_loglam / dloglam
    bound = -1.0 / (sigma_pix ** 2)
    frac = np.mean(curvature < bound) * 100.0
    return bound, frac


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    timesteps = int(ckpt.get("timesteps", 1000)) if args.timesteps is None else args.timesteps

    with h5py.File(args.h5, "r") as h5:
        n_pix = int(h5["flux_raw"].shape[-1])

    model = UNet1D(in_ch=1, base_ch=64, depth=4, time_dim=256).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    real_flux = load_real_samples(args.h5, args.n_samples, args.seed)
    gen_flux = ddpm_sample(
        model, real_flux.shape[0], n_pix, timesteps, args.batch_size, device
    )

    real_flat = real_flux.reshape(real_flux.shape[0], -1)
    gen_flat = gen_flux.reshape(gen_flux.shape[0], -1)

    # Flux PDF
    all_vals = np.concatenate([real_flat.ravel(), gen_flat.ravel()])
    vmin, vmax = np.percentile(all_vals, [0.5, 99.5])
    bins = np.linspace(vmin, vmax, 80)

    # Power spectrum
    k, pk_real = compute_power_spectrum(real_flat)
    _, pk_gen = compute_power_spectrum(gen_flat)

    # Curvature
    curv_real = curvature_stat(real_flat)
    curv_gen = curvature_stat(gen_flat)
    bound, frac_real = lower_bound_fraction(curv_real, n_pix)
    _, frac_gen = lower_bound_fraction(curv_gen, n_pix)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].hist(
        real_flat.ravel(), bins=bins, density=True, alpha=0.6, label="Real"
    )
    axes[0].hist(
        gen_flat.ravel(), bins=bins, density=True, alpha=0.6, label="Gen"
    )
    axes[0].set_title("Flux PDF")
    axes[0].set_xlabel("Flux")
    axes[0].set_ylabel("Probability Density")
    axes[0].legend()

    axes[1].plot(k, pk_real, label="Real")
    axes[1].plot(k, pk_gen, label="Gen")
    axes[1].set_title("Power Spectrum")
    axes[1].set_xlabel("k (pixel$^{-1}$)")
    axes[1].set_ylabel("P(k)")
    axes[1].set_yscale("log")
    axes[1].legend()

    curv_all = np.concatenate([curv_real.ravel(), curv_gen.ravel()])
    cmin, cmax = np.percentile(curv_all, [0.5, 99.5])
    cbins = np.linspace(cmin, cmax, 80)
    axes[2].hist(
        curv_real.ravel(), bins=cbins, density=True, alpha=0.6, label="Real"
    )
    axes[2].hist(
        curv_gen.ravel(), bins=cbins, density=True, alpha=0.6, label="Gen"
    )
    axes[2].axvline(bound, color="k", linestyle="--", linewidth=1.0, label="Lower bound")
    axes[2].set_title("Curvature D = d2/dx2 log(F)")
    axes[2].set_xlabel("D")
    axes[2].set_ylabel("Probability Density")
    axes[2].legend()
    axes[2].text(
        0.02,
        0.98,
        f"Violations (D < bound):\nReal {frac_real:.2f}% | Gen {frac_gen:.2f}%",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"),
    )

    fig.suptitle(f"Validation Report | lower bound D_min={bound:.3e}", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"[INFO] Saved report: {args.out}")


if __name__ == "__main__":
    main()
