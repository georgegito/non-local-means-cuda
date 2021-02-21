#ifndef __FILTERING_H__
#define __FILTERING_H__

#include <utils.hpp>

double filterPixel( double* image, 
                    double* _distances, 
                    double* _weights, 
                    int n, 
                    int patchSize, 
                    int pixelRow, 
                    int pixelCol, 
                    double sigma )
{
    double res = 0;
    double sumW = 0;                    // sumW is the Z(i) of w(i, j) formula
    double dist;
    std::vector<double> weights(n * n);
    int patchRowStart = pixelRow - patchSize / 2;
    int patchColStart = pixelCol - patchSize / 2;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dist = util::computePatchDistance(  image, 
                                                _distances, 
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

std::vector<double> filterImage( double* image, 
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

#endif // __FILTERING_H__