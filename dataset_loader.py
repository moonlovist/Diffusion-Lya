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
        self.h5_file = None
        with h5py.File(self.h5_path, "r") as h5:
            self._len = int(h5["flux_raw"].shape[0])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(self.h5_path)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", swmr=True)
        h5 = self.h5_file

        clean = np.asarray(h5["flux_raw"][idx], dtype=np.float32)
        noisy = np.asarray(h5["flux_noisy"][idx], dtype=np.float32)
        sigma = np.asarray(h5["sigma_map"][idx], dtype=np.float32)
        scale = np.float32(h5["scale_factor"][idx])

        clean_norm = clean / scale

        return {
            "clean": torch.from_numpy(clean_norm),
            "noisy": torch.from_numpy(noisy),
            "sigma": torch.from_numpy(sigma),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }


def split_train_val(dataset, val_frac=0.1, seed=42):
    n_total = len(dataset)
    n_val = int(round(n_total * val_frac))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=gen)
