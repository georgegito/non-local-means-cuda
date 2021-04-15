# Non Local Means - CUDA

Implementation of **Non Local Means** algorithm for image denoising.

* **version 1**: CPU sequential in C++
* **version 2**: GPU parallel in CUDA
* **version 3**: GPU parallel using shared memory in CUDA

## Build instructions

**Requierements:** nvcc (Nvidia CUDA Compiler)

Set parameters in Makefile and
```bash
make
```
or 

```bash
mkdir -p data/out
nvcc -o build/main -I./include src/main.cu -O3
```

and run with

```bash
./build/main $(imageNum) $(patchSize) $(filterSigma) $(patchSigma) $(useGpu) $(useSharedMem)
```

## Instructions to see the filtered images

Run the script `matlab/show_image.m` in MATLAB and all the output images of folder `data/out` will show up.

Original and noisy images can be found in `data/images`.
