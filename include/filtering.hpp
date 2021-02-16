#ifndef __FILTERING_H__
#define __FILTERING_H__

#include <utils.hpp>

double filterPixel(std::vector<std::vector<int>> image, int n, int patchSize, int pixelRow, int pixelCol, double sigma)
{
    double res = 0;
    double sumW = 0;                    // sumW is the Z(i) of w(i, j) formula
    double dist;
    std::vector<double> weights(n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << "pixel (" << i << ", " << j << ")" << std::endl;
            dist = util::computeEuclideanDistance(image, n, patchSize, pixelRow, pixelCol, i, j);
            std::cout << "distance = " << dist << std::endl;
            weights[i * n + j] = util::computeWeight(dist, sigma);
            std::cout << "weight = " << weights[i * n + j] << std::endl << std::endl;
            sumW += weights[i * n + j];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j++) {
            res += (weights[i * n + j] / sumW) * image[i][j];
        }
    }

    return res;
}

#endif // __FILTERING_H__