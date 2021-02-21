#ifndef __CUDAFILTERING_CUH__
#define __CUDAFILTERING_CUH__

#include <utils.cuh>

__global__ void computeWeights( double *image,
    double *_weights,
    int n,
    int patchSize,
    int patchRowStart,
    int patchColStart,
    double sigma,
    double *weights)
{

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n * n){
        return;
    }
    int row = index / n;
    int col = index % n;

    double dist = util::computePatchDistance(  image, 
                        _weights, 
                        n, 
                        patchSize, 
                        patchRowStart, 
                        patchColStart, 
                        row - patchSize / 2, 
                        col - patchSize / 2  );
    weights[index] =  util::computeWeight(dist, sigma);

}

double cudaFilterPixel( double* image, 
                    double* _weights, 
                    int n, 
                    int patchSize, 
                    int pixelRow, 
                    int pixelCol, 
                    double sigma )
{
    double res = 0;
    double sumW = 0;                    // sumW is the Z(i) of w(i, j) formula

    std::vector<double> weights(n * n);
    int patchRowStart = pixelRow - patchSize / 2;
    int patchColStart = pixelCol - patchSize / 2;

    double *d_image, *d_insideWeights, *d_weights;
    int size_image = n * n * sizeof(double);
    int size_insideWeights = patchSize * patchSize * sizeof(double);

    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_image, size_image);
    cudaMalloc((void **)&d_insideWeights, size_insideWeights);
    cudaMalloc((void **)&d_weights, size_image);
    
    cudaMemcpy(d_image, image, size_image, cudaMemcpyHostToDevice);
    cudaMemcpy(d_insideWeights, _weights, size_insideWeights, cudaMemcpyHostToDevice);

    computeWeights<<<n,n>>>(    d_image,
                                d_insideWeights,
                                n,
                                patchSize,
                                patchRowStart,
                                patchColStart,
                                sigma,
                                d_weights);

    cudaMemcpy(weights.data(), d_weights, size_image, cudaMemcpyDeviceToHost);
    

    cudaFree(d_image); cudaFree(d_insideWeights); cudaFree(d_weights);

    for(int i = 0; i < n * n; i++){
        sumW += weights[i];
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res += (weights[i * n + j] / sumW) * image[i * n + j];
        }
    }

    return res;
}

std::vector<double> cudaFilterImage( double* image, 
                                 int n, 
                                 int patchSize,  
                                 double patchSigma,
                                 double filterSigma )
{
    std::vector<double> res(n * n);
    // std::vector<double> _distances = util::computeDistanceMatrix(image, n);
    double* _weights = util::computeInsideWeights(patchSize, patchSigma).data();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i * n + j] = cudaFilterPixel(image, _weights, n, patchSize, i, j, filterSigma);
        }
    }

    return res;
}

#endif // __CUDAFILTERING_CUH__