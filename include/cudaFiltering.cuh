#ifndef __CUDAFILTERING_CUH__
#define __CUDAFILTERING_CUH__

#include <utils.hpp>

__global__ void filterKernel( double* image, 
                              double* _distances, 
                              double* _weights, 
                              int n, 
                              int patchSize, 
                              int pixelRow, 
                              int pixelCol, 
                              double sigma )
{

/* ------------------------------ filter pixel ------------------------------ */

    // int i = blockIdx.x*blockDim.x + threadIdx.x;
    // if (i < n) y[i] = a*x[i] + y[i];
}

std::vector<double> cudaFilterImage( double* image, 
                                     int n, 
                                     int patchSize,  
                                     double patchSigma,
                                     double filterSigma )
{
    std::vector<double> res(n * n);
    // std::vector<double> _distances = util::computeDistanceMatrix(image, n);
    double* _distances;
    double* _weights = util::computeInsideWeights(patchSize, patchSigma).data();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i * n + j] = filterPixel(image, _distances, _weights, n, patchSize, i, j, filterSigma);
        }
    }

    return res;
}

#endif // __CUDAFILTERING_CUH__