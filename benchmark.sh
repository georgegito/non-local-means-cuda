#!/bin/bash
#SBATCH --job-name=non-local-means
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00

module load gcc
module load cuda

nvidia-smi
make