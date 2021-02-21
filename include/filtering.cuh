#ifndef __FILTERING_CUH__
#define __FILTERING_CUH__

#include <utils.cuh>

float filterPixel( float * image, 
                   float * _weights, 
                   int n, 
                   int patchSize, 
                   int pixelRow, 
                   int pixelCol, 
                   float sigma )
{
    float res = 0;
    float sumW = 0;                    // sumW is the Z(i) of w(i, j) formula
    float dist;
    std::vector<float> weights(n * n);
    int patchRowStart = pixelRow - patchSize / 2;
    int patchColStart = pixelCol - patchSize / 2;

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
            weights[i * n + j] = util::computeWeight(dist, sigma);
            sumW += weights[i * n + j];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res += (weights[i * n + j] / sumW) * image[i * n + j];
        }
    }

    return res;
}

std::vector<float > filterImage( float * image, 
                                 int n, 
                                 int patchSize,  
                                 float patchSigma,
                                 float filterSigma )
{
    std::vector<float > res(n * n);
    // std::vector<float > _distances = util::computeDistanceMatrix(image, n);
    float * _weights = util::computeInsideWeights(patchSize, patchSigma).data();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i * n + j] = filterPixel(image, _weights, n, patchSize, i, j, filterSigma);
        }
    }

    return res;
}

#endif // __FILTERING_CUH__