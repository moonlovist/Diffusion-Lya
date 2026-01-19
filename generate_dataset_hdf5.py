#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a large QSO dataset into a single HDF5 file.

Requirements:
  - simqso available (for sqbase.fixed_R_dispersion)

Output:
  - dataset_v1.h5 in $SCRATCH (or current directory if SCRATCH unset)
  - datasets (fixed length 2000, interpolated to TARGET_WAVE_GRID):
      flux_raw   (N, 1, 2000)
      flux_noisy (N, 1, 2000)
      sigma_map  (N, 1, 2000)
      meta/z, meta/snr, meta/dla_logNHI
      scale_factor (N,)
"""

import os
import multiprocessing as mp

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

import sys
from astropy.cosmology import Planck13
from astropy import constants as const
from scipy.special import wofz

# If simqso is in a custom path, set SIMQSO_PATH or edit here.
SIMQSO_PATH = os.environ.get(
    "SIMQSO_PATH", "/global/cfs/cdirs/desi/users/tingtan/CSST_DLA/simqso/desisim/"
)
if SIMQSO_PATH and SIMQSO_PATH not in sys.path:
    sys.path.append(SIMQSO_PATH)

from simqso.sqgrids import generateQlfPoints
from simqso import sqbase
from simqso.sqrun import buildSpectraBulk
from simqso.sqmodels import BOSS_DR9_PLEpivot, get_BossDr9_model_vars

N_SAMPLES = 50_000
CHUNK_SIZE = 1000
R_OUT = 241.0
L_MIN = 2000.0
L_MAX = 8000.0
NPIX = 2000
MIN_SIGMA = 0.02
SNR_MIN = 2.0
SNR_MAX = 10.0
DLA_LOGNHI_MIN = 20.3
DLA_LOGNHI_MAX = 22.0

TARGET_WAVE_GRID = np.linspace(2000.0, 8000.0, 2000)

_WAVE = None

c_cgs = const.c.to("cm/s").value


def voigt_wofz(vin, a):
    return wofz(vin + 1j * a).real


def voigt_tau(wave_cm, par):
    cold = 10.0 ** par[0]
    zp1 = par[1] + 1.0
    nujk = c_cgs / par[3]
    dnu = par[2] / par[3]
    avoigt = par[5] / (4 * np.pi * dnu)
    uvoigt = ((c_cgs / (wave_cm / zp1)) - nujk) / dnu
    cne = 0.014971475 * cold * par[4]
    tau = cne * voigt_wofz(uvoigt, avoigt) / dnu
    return tau


def dla_spec(wave_ang, dlas):
    flya = 0.4164
    gamma_lya = 6.265e8
    lyacm = 1215.6700 / 1e8
    wavecm = wave_ang / 1e8

    tau = np.zeros(wave_ang.size, dtype=float)
    for dla in dlas:
        par = [
            dla["N"],
            dla["z"],
            30.0 * 1e5,
            lyacm,
            flya,
            gamma_lya,
        ]
        tau += voigt_tau(wavecm, par)
    return np.exp(-tau)


def insert_dlas3(wave_ang, z_dla, logNHI):
    dlas = [dict(z=z_dla, N=logNHI, dlaid=0)]
    dla_model = dla_spec(wave_ang, dlas)
    return dlas, dla_model


def generate_qso_spectrum(wave, z_qso, qlf_seed=12345, grid_seed=67890, forest_seed=192837465):
    kcorr = sqbase.ContinuumKCorr("DECam-r", 1450, effWaveBand="SDSS-r")
    qsos = generateQlfPoints(
        BOSS_DR9_PLEpivot(cosmo=Planck13),
        (19, 22),
        (1.7, 3.8),
        kcorr=kcorr,
        zin=[z_qso],
        qlfseed=qlf_seed,
        gridseed=grid_seed,
    )
    sedVars = get_BossDr9_model_vars(qsos, wave, 0, forestseed=forest_seed, verbose=0)
    qsos.addVars(sedVars)
    qsos.loadPhotoMap([("DECam", "DECaLS"), ("WISE", "AllWISE")])
    _, spectra = buildSpectraBulk(wave, qsos, saveSpectra=True, maxIter=3, verbose=0)
    if spectra is None:
        raise RuntimeError("buildSpectraBulk returned None.")
    return spectra[0], qsos


def snr_profile_from_mean(wave, snr_mean):
    """Monotonic SNR profile with mean == snr_mean."""
    lam_min = float(np.min(wave))
    lam_max = float(np.max(wave))
    x = (wave - lam_min) / (lam_max - lam_min)
    alpha = 1.5
    base = 1.0 + alpha * x
    snr_lambda = snr_mean * base / np.mean(base)
    return snr_lambda


def filter_like_pipeline(wave, flux, snr_mean, rng):
    """
    Degrade resolution + apply noise.
    Returns: flux_raw, flux_noisy, sigma_map, scale_factor
    """
    loglam = np.log(wave)
    dloglam = np.mean(np.diff(loglam))
    sigma_loglam = 1.0 / (2.355 * R_OUT)
    sigma_pix = sigma_loglam / dloglam

    degraded = gaussian_filter1d(flux.astype(np.float64), sigma=sigma_pix, mode="reflect")

    scale = np.percentile(np.abs(degraded), 95)
    if (not np.isfinite(scale)) or (scale <= 0):
        scale = max(np.mean(np.abs(degraded)), 1.0)
    f_norm = degraded / scale

    snr_lambda = snr_profile_from_mean(wave, snr_mean=snr_mean)
    sigma_noise = np.sqrt(MIN_SIGMA**2 + (np.abs(f_norm) / snr_lambda) ** 2)
    noise = rng.normal(0.0, sigma_noise, size=wave.size)

    noisy = np.clip(f_norm + noise, 0.0, None)

    flux_raw = degraded.astype(np.float32)
    flux_noisy = (noisy * scale).astype(np.float32)
    sigma_map = (sigma_noise * scale).astype(np.float32)
    return flux_raw, flux_noisy, sigma_map, float(scale)


def _init_worker():
    global _WAVE
    if _WAVE is not None:
        return
    _WAVE = sqbase.fixed_R_dispersion(L_MIN, L_MAX, NPIX)


def _interp_to_target(wave, arr):
    f = interp1d(wave, arr, kind="linear", fill_value="extrapolate")
    return f(TARGET_WAVE_GRID).astype(np.float32)


def _worker(params):
    z_qso, snr_mean, has_dla, z_dla, logNHI, seed = params
    rng = np.random.default_rng(seed)

    wave = _WAVE
    spec_qso, _ = generate_qso_spectrum(wave, z_qso)

    if has_dla:
        _, dla_model = insert_dlas3(wave, z_dla, logNHI)
        spec_in = spec_qso * dla_model
    else:
        spec_in = spec_qso
        logNHI = np.nan

    flux_raw, flux_noisy, sigma_map, scale = filter_like_pipeline(
        wave, spec_in, snr_mean=snr_mean, rng=rng
    )

    flux_raw = _interp_to_target(wave, flux_raw)
    flux_noisy = _interp_to_target(wave, flux_noisy)
    sigma_map = _interp_to_target(wave, sigma_map)

    return flux_raw, flux_noisy, sigma_map, scale, z_qso, snr_mean, logNHI


def _log_uniform(rng, low, high, size):
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Generate QSO spectra into one HDF5 file.")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--r-out", type=float, default=R_OUT)
    p.add_argument("--l-min", type=float, default=L_MIN)
    p.add_argument("--l-max", type=float, default=L_MAX)
    p.add_argument("--npix", type=int, default=NPIX)
    p.add_argument("--min-sigma", type=float, default=MIN_SIGMA)
    p.add_argument("--snr-min", type=float, default=SNR_MIN)
    p.add_argument("--snr-max", type=float, default=SNR_MAX)
    p.add_argument("--dla-lognhi-min", type=float, default=DLA_LOGNHI_MIN)
    p.add_argument("--dla-lognhi-max", type=float, default=DLA_LOGNHI_MAX)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--output-file", type=str, default=None)
    p.add_argument("--catalog-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    scratch = os.environ.get("SCRATCH", "/pscratch/sd/t/tanting/diffusion")
    if args.output_dir:
        scratch = args.output_dir
    out_path = os.path.join(scratch, "dataset_v1.h5")
    if args.output_file:
        out_path = args.output_file
        scratch = os.path.dirname(out_path) or "."
    os.makedirs(scratch, exist_ok=True)

    master_seed = 20251031
    rng = np.random.default_rng(master_seed)

    global R_OUT, L_MIN, L_MAX, NPIX, MIN_SIGMA, SNR_MIN, SNR_MAX, DLA_LOGNHI_MIN, DLA_LOGNHI_MAX
    global N_SAMPLES, CHUNK_SIZE
    N_SAMPLES = int(args.n_samples)
    CHUNK_SIZE = int(args.chunk_size)
    R_OUT = float(args.r_out)
    L_MIN = float(args.l_min)
    L_MAX = float(args.l_max)
    NPIX = int(args.npix)
    MIN_SIGMA = float(args.min_sigma)
    SNR_MIN = float(args.snr_min)
    SNR_MAX = float(args.snr_max)
    DLA_LOGNHI_MIN = float(args.dla_lognhi_min)
    DLA_LOGNHI_MAX = float(args.dla_lognhi_max)

    wave_size = TARGET_WAVE_GRID.size

    z_qso = rng.uniform(2.0, 3.5, size=N_SAMPLES)
    snr_mean = _log_uniform(rng, SNR_MIN, SNR_MAX, size=N_SAMPLES)
    has_dla = rng.random(size=N_SAMPLES) < 0.5

    z_dla = np.empty(N_SAMPLES, dtype=np.float64)
    logNHI = np.empty(N_SAMPLES, dtype=np.float64)
    for i in range(N_SAMPLES):
        if has_dla[i]:
            z_max = max(2.0, z_qso[i])
            z_dla[i] = rng.uniform(2.0, z_max)
            logNHI[i] = rng.uniform(DLA_LOGNHI_MIN, DLA_LOGNHI_MAX)
        else:
            z_dla[i] = np.nan
            logNHI[i] = np.nan

    seeds = rng.integers(0, 2**32 - 1, size=N_SAMPLES, dtype=np.uint32)
    params = [
        (float(z_qso[i]), float(snr_mean[i]), bool(has_dla[i]), float(z_dla[i]), float(logNHI[i]), int(seeds[i]))
        for i in range(N_SAMPLES)
    ]

    nproc = os.cpu_count() or 1

    catalog_path = None
    catalog_fh = None
    if args.catalog_dir:
        os.makedirs(args.catalog_dir, exist_ok=True)
        catalog_path = os.path.join(args.catalog_dir, "catalog.csv")
        catalog_fh = open(catalog_path, "w", encoding="ascii", newline="")
        catalog_fh.write("index,z,snr,dla_logNHI,scale_factor\n")

    with h5py.File(out_path, "w") as h5:
        flux_raw_ds = h5.create_dataset(
            "flux_raw",
            shape=(N_SAMPLES, 1, wave_size),
            dtype="f4",
            chunks=(CHUNK_SIZE, 1, wave_size),
        )
        flux_noisy_ds = h5.create_dataset(
            "flux_noisy",
            shape=(N_SAMPLES, 1, wave_size),
            dtype="f4",
            chunks=(CHUNK_SIZE, 1, wave_size),
        )
        sigma_map_ds = h5.create_dataset(
            "sigma_map",
            shape=(N_SAMPLES, 1, wave_size),
            dtype="f4",
            chunks=(CHUNK_SIZE, 1, wave_size),
        )
        meta_grp = h5.create_group("meta")
        z_ds = meta_grp.create_dataset("z", shape=(N_SAMPLES,), dtype="f4")
        snr_ds = meta_grp.create_dataset("snr", shape=(N_SAMPLES,), dtype="f4")
        dla_ds = meta_grp.create_dataset("dla_logNHI", shape=(N_SAMPLES,), dtype="f4")
        scale_ds = h5.create_dataset("scale_factor", shape=(N_SAMPLES,), dtype="f4")

        buffer = []
        written = 0

        with mp.Pool(processes=nproc, initializer=_init_worker) as pool:
            for result in pool.imap_unordered(_worker, params):
                buffer.append(result)
                if len(buffer) >= CHUNK_SIZE:
                    raw = np.stack([r[0] for r in buffer], axis=0)
                    noisy = np.stack([r[1] for r in buffer], axis=0)
                    sig = np.stack([r[2] for r in buffer], axis=0)
                    scales = np.array([r[3] for r in buffer], dtype=np.float32)
                    zs = np.array([r[4] for r in buffer], dtype=np.float32)
                    snrs = np.array([r[5] for r in buffer], dtype=np.float32)
                    dlas = np.array([r[6] for r in buffer], dtype=np.float32)

                    end = written + raw.shape[0]
                    flux_raw_ds[written:end, 0, :] = raw
                    flux_noisy_ds[written:end, 0, :] = noisy
                    sigma_map_ds[written:end, 0, :] = sig
                    scale_ds[written:end] = scales
                    z_ds[written:end] = zs
                    snr_ds[written:end] = snrs
                    dla_ds[written:end] = dlas

                    if catalog_fh is not None:
                        for i in range(written, end):
                            j = i - written
                            catalog_fh.write(
                                f"{i},{zs[j]:.6f},{snrs[j]:.6f},{dlas[j]:.6f},{scales[j]:.6f}\n"
                            )

                    written = end
                    print(f"[INFO] Wrote {written}/{N_SAMPLES} spectra")
                    buffer = []

        if buffer:
            raw = np.stack([r[0] for r in buffer], axis=0)
            noisy = np.stack([r[1] for r in buffer], axis=0)
            sig = np.stack([r[2] for r in buffer], axis=0)
            scales = np.array([r[3] for r in buffer], dtype=np.float32)
            zs = np.array([r[4] for r in buffer], dtype=np.float32)
            snrs = np.array([r[5] for r in buffer], dtype=np.float32)
            dlas = np.array([r[6] for r in buffer], dtype=np.float32)

            end = written + raw.shape[0]
            flux_raw_ds[written:end, 0, :] = raw
            flux_noisy_ds[written:end, 0, :] = noisy
            sigma_map_ds[written:end, 0, :] = sig
            scale_ds[written:end] = scales
            z_ds[written:end] = zs
            snr_ds[written:end] = snrs
            dla_ds[written:end] = dlas

            if catalog_fh is not None:
                for i in range(written, end):
                    j = i - written
                    catalog_fh.write(
                        f"{i},{zs[j]:.6f},{snrs[j]:.6f},{dlas[j]:.6f},{scales[j]:.6f}\n"
                    )

            written = end
            print(f"[INFO] Wrote {written}/{N_SAMPLES} spectra")

    if catalog_fh is not None:
        catalog_fh.close()
        print(f"[INFO] Wrote catalog: {catalog_path}")
    print(f"[INFO] Finished. Output: {out_path}")


if __name__ == "__main__":
    main()
