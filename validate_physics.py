#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm

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
    p.add_argument("--out", type=str, default="validation_outputs.npz")
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
    curv = np.zeros_like(log_flux)
    curv[:, 1:-1] = log_flux[:, 2:] + log_flux[:, :-2] - 2.0 * log_flux[:, 1:-1]
    curv[:, 0] = np.nan
    curv[:, -1] = np.nan
    return curv


def lower_bound_fraction(curvature, n_pix, r_out=241.0, l_min=2000.0, l_max=8000.0):
    wave = np.linspace(l_min, l_max, n_pix)
    dlambda = np.gradient(wave)
    sigma_angstrom = wave / (2.355 * r_out)
    sigma_pix_arr = sigma_angstrom / dlambda
    bound_arr = -1.0 / (sigma_pix_arr ** 2)
    valid = ~np.isnan(curvature)
    frac = np.mean((curvature < bound_arr[None, :]) & valid) * 100.0
    return bound_arr, frac


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

    # Power spectrum
    k, pk_real = compute_power_spectrum(real_flat)
    _, pk_gen = compute_power_spectrum(gen_flat)

    # Curvature
    curv_real = curvature_stat(real_flat)
    curv_gen = curvature_stat(gen_flat)
    bound_arr, frac_real = lower_bound_fraction(curv_real, n_pix)
    _, frac_gen = lower_bound_fraction(curv_gen, n_pix)
    wave = np.linspace(2000.0, 8000.0, n_pix)

    np.savez(
        args.out,
        x_real=real_flat.astype(np.float32),
        x_gen=gen_flat.astype(np.float32),
        wave=wave.astype(np.float32),
        k=k.astype(np.float32),
        pk_real=pk_real.astype(np.float32),
        pk_gen=pk_gen.astype(np.float32),
        curv_real=curv_real.astype(np.float32),
        curv_gen=curv_gen.astype(np.float32),
        bound=bound_arr.astype(np.float32),
        frac_real=np.array([frac_real], dtype=np.float32),
        frac_gen=np.array([frac_gen], dtype=np.float32),
    )
    print(f"[INFO] Saved outputs: {args.out}")


if __name__ == "__main__":
    main()
