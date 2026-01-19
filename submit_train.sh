#!/bin/bash
#SBATCH -A regular
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -t 08:00:00
#SBATCH -J lya_ddpm

set -euo pipefail

module load pytorch/2.0.1

python train_ddpm.py \
  --h5 /pscratch/sd/t/tanting/diffusion/dataset_v1.h5 \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --timesteps 1000 \
  --num-workers 4
