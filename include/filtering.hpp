#ifndef __FILTERING_H__
#define __FILTERING_H__

#include <utils.hpp>

double filterPixel(std::vector<std::vector<int>> image, int n, int patchSize, int pixelRow, int pixelCol, double sigma)
{
    double res = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double dist = util::computeEuclideanDistance(image, n, patchSize, pixelRow, pixelCol, i, j);
            double w = util::computeWeight(dist, sigma, n * n);
            // std::cout << w << std::endl;
            res += w * image[i][j];
        }
    }

    return res;
}

#endif // __FILTERING_H__