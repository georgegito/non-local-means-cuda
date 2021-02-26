#!/bin/bash
#SBATCH --job-name=non-local-means
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00

module load gcc
module load cuda
nvidia-smi

#imageNum={house:0, flower:1, lena:2}
imageNum=0

#parameters
patchSize=3
filterSigma=(0.01 0.05 0.1)
patchSigma=(0.8)
useGpu=0
useSharedMem=0

#make build dir
mkdir -p build

#compile
nvcc -o build/main -I./include src/main.cu -O3

for i in ${filterSigma[@]}; do
    for j in ${patchSigma[@]}; do
        #run
        ./build/main $imageNum $patchSize $i $j $useGpu $useSharedMem 
    done
done


