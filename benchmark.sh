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
imageNum=1

#parameters
patchSize=5
filterSigma=(0.04 0.06 0.08 0.1 0.12 0.14)
patchSigma=(0.4 0.6 0.8 1 1.2 1.4)
useGpu=1
useSharedMem=1

#compile
mkdir -p build
mkdir -lude src/main.cu -O3

for i in ${filterSigma[@]}; do
    for j in ${patchSigma[@]}; do
        #run
        ./build/main $imageNum $patchSize $i $j $useGpu $useSharedMem 
    done
done


