#ifndef __CUDAFILTERINGGLOBALMEM_CUH__
#define __CUDAFILTERINGGLOBALMEM_CUH__

#include <utils.cuh>

namespace gpuGlobalMem {

__global__ void filterPixel(float * image, 
                                float * _weights, 
                                int n, 
                                int patchSize, 
                                float sigma,
                                float *filteredImage)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (index >= n*n){
        return;
    }

    int pixelRow = blockIdx.x;
    int pixelCol = threadIdx.x;

    float res = 0;
    float sumW = 0;                    // sumW is the Z(i) of w(i, j) formula
    float dist;
    float w;
    int patchRowStart = pixelRow - patchSize / 2;
    int patchColStart = pixelCol - patchSize / 2;

    __syncthreads();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dist = util::computePatchDistance(  image,  
                                                _weights, 
                                                n, 
                                                patchSize, 
                                                patchRowStart, 
                                                patchColStart, 
                                                i - patchSize / 2, 
                                                j - patchSize / 2  );
            w = util::computeWeight(dist, sigma);
            sumW += w;
            res += w * image[i * n + j];
        }
    }
    res = res / sumW;

    filteredImage[index] = res;
    }

std::vector<float> filterImage( float * image, 
                                int n, 
                                int patchSize,  
                                float patchSigma,
                                float filterSigma )
{
    std::vector<float> res(n * n);
    float * _weights = util::computeInsideWeights(patchSize, patchSigma);

    int size_image = n * n * sizeof(float);
    int size_weights = patchSize * patchSize * sizeof(float );

    float *d_image, *d_weights, *d_res;

    cudaMalloc((void **)&d_image, size_image);
    cudaMalloc((void **)&d_weights, size_weights);
    cudaMalloc((void **)&d_res, size_image);

    cudaMemcpy(d_image, image, size_image, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, _weights, size_weights, cudaMemcpyHostToDevice);

    filterPixel<<<n,n>>>(d_image, d_weights, n, patchSize, filterSigma, d_res);
    
    cudaMemcpy(res.data(), d_res, size_image, cudaMemcpyDeviceToHost);

    cudaFree(d_image); 
    cudaFree(d_weights); 
    cudaFree(d_res);

    return res;
}

} // namespace gpuGlobalMem



#endif // __CUDAFILTERINGGLOBALMEM_CUH__