#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, device="cpu"):
    return torch.linspace(beta_start, beta_end, timesteps, device=device)


def ddpm_step(x_t, eps, t, betas, alphas, alpha_bars):
    """
    Standard DDPM reverse step using predicted noise eps.
    """
    bsz = x_t.shape[0]
    beta_t = betas[t].view(bsz, 1, 1)
    alpha_t = alphas[t].view(bsz, 1, 1)
    alpha_bar_t = alpha_bars[t].view(bsz, 1, 1)

    coef = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
    mean = (x_t - coef * eps) / torch.sqrt(alpha_t)

    alpha_bar_prev = alpha_bars[torch.clamp(t - 1, min=0)].view(bsz, 1, 1)
    var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    noise = torch.randn_like(x_t)
    mask = (t > 0).float().view(bsz, 1, 1)
    return mean + mask * torch.sqrt(var) * noise


def dps_sample(
    model,
    y,
    sigma,
    scale,
    timesteps=1000,
    step_size=1e-2,
    device=None,
):
    """
    Diffusion Posterior Sampling (DPS).

    Args:
      y: noisy observation, shape (B, 1, L)
      sigma: noise map, shape (B, 1, L)
      scale: normalization factor, shape (B,) or (B, 1, 1)
    """
    if device is None:
        device = y.device

    y = y.to(device)
    sigma = sigma.to(device)
    if scale.ndim == 1:
        scale = scale[:, None, None]
    scale = scale.to(device)

    betas = linear_beta_schedule(timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    x_t = torch.randn_like(y, device=device)

    model.eval()
    for t_idx in range(timesteps - 1, -1, -1):
        t = torch.full((x_t.shape[0],), t_idx, device=device, dtype=torch.long)

        with torch.no_grad():
            eps = model(x_t, t.float())

        alpha_bar_t = alpha_bars[t].view(x_t.shape[0], 1, 1)
        x_t = x_t.detach()
        x_t.requires_grad_(True)
        x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

        # Negative log-likelihood (up to constant)
        resid = y - scale * x0_hat
        likelihood_loss = torch.sum((resid ** 2) / (sigma ** 2))
        g = torch.autograd.grad(likelihood_loss, x_t)[0]

        with torch.no_grad():
            x_prev = ddpm_step(x_t, eps, t, betas, alphas, alpha_bars)
            x_prev = x_prev - step_size * g
            x_t = x_prev

    return x_t
