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
    float w;
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
            w = util::computeWeight(dist, sigma);
            sumW += w;
            res += w * image[i * n + j];
        }
    }
    res = res / sumW;

    return res;
}

std::vector<float> filterImage( float * image, 
                                 int n, 
                                 int patchSize,  
                                 float patchSigma,
                                 float filterSigma )
{
    std::vector<float> res(n * n);
    float * _weights = util::computeInsideWeights(patchSize, patchSigma);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i * n + j] = filterPixel(image, _weights, n, patchSize, i, j, filterSigma);
        }
    }

    return res;
}

#endif // __FILTERING_CUH__