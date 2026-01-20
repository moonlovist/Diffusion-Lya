#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, depth=4, time_dim=256, groups=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.in_conv = nn.Conv1d(in_ch, base_ch, kernel_size=3, padding=1)

        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_c = base_ch
        for i, out_c in enumerate(chs):
            self.down_blocks.append(ResBlock1D(in_c, out_c, time_dim, groups=groups))
            in_c = out_c
            if i < depth - 1:
                self.downsamples.append(
                    nn.Conv1d(in_c, in_c, kernel_size=4, stride=2, padding=1)
                )

        self.mid1 = ResBlock1D(in_c, in_c, time_dim, groups=groups)
        self.mid2 = ResBlock1D(in_c, in_c, time_dim, groups=groups)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for out_c in reversed(chs[:-1]):
            self.upsamples.append(nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1))
            self.up_blocks.append(ResBlock1D(out_c * 2, out_c, time_dim, groups=groups))
            in_c = out_c

        self.out_conv = nn.Conv1d(base_ch, in_ch, kernel_size=3, padding=1)

    def _match_length(self, x, target_len):
        if x.shape[-1] == target_len:
            return x
        if x.shape[-1] > target_len:
            return x[..., :target_len]
        pad = target_len - x.shape[-1]
        return F.pad(x, (0, pad))

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)

        skips = []
        for i, block in enumerate(self.down_blocks):
            h = block(h, t_emb)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for i, up in enumerate(self.upsamples):
            h = up(h)
            skip = skips[-(i + 2)]
            h = self._match_length(h, skip.shape[-1])
            h = torch.cat([h, skip], dim=1)
            h = self.up_blocks[i](h, t_emb)

        h = self._match_length(h, x.shape[-1])
        return self.out_conv(h)
