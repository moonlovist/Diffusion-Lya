#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class LyaDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self._h5 = None
        self._len = None
        self._init_len()

    def _init_len(self):
        with h5py.File(self.h5_path, "r") as h5:
            self._len = int(h5["flux_raw"].shape[0])

    def _ensure_open(self):
        if self._h5 is None:
            if not os.path.exists(self.h5_path):
                raise FileNotFoundError(self.h5_path)
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._ensure_open()
        x0 = self._h5["flux_raw"][idx]       # (1, L)
        y = self._h5["flux_noisy"][idx]      # (1, L)
        sigma = self._h5["sigma_map"][idx]   # (1, L)
        scale = self._h5["scale_factor"][idx]

        x0 = np.asarray(x0, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        sigma = np.asarray(sigma, dtype=np.float32)
        scale = np.float32(scale)

        x_norm = x0 / scale

        return {
            "x_0": torch.from_numpy(x_norm),
            "y": torch.from_numpy(y),
            "sigma": torch.from_numpy(sigma),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def split_train_val(dataset, val_frac=0.1, seed=42):
    n_total = len(dataset)
    n_val = int(round(n_total * val_frac))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=gen)
