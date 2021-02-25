#!/bin/bash
#SBATCH --job-name=non-local-means
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00

module load gcc
module load cuda
nvidia-smi

#imageNum={house:0, flower:1, lena:2}
imageNum=0

#parameters
patchSize=5
filterSigma=0.06
patchSigma=0.8
useGpu=0
useSharedMem=0

#compile
nvcc -o build/main -I./include src/main.cu -O3

#run
./build/main $imageNum $patchSize $filterSigma $patchSigma $useGpu $useSharedMem