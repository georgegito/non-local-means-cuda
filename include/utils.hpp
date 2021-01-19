#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <vector>
#include <cmath>

namespace util {

bool isInBounds(int n, int x, int y) 
{
    return x >=0 && x < n && y >= 0 && y < n;
}

double computeEuclideanDistance(std::vector<std::vector<int>> image, int n, int patchSize, int p1_row, int p1_col, int p2_row, int p2_col) 
{
    int p1_rowStart = p1_row - patchSize / 2;   // TODO avoid multiple computations for p1
    int p1_colStart = p1_col - patchSize / 2;
    int p2_rowStart = p2_row - patchSize / 2;
    int p2_colStart = p2_col - patchSize / 2;
    double ans = 0;

    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            if (isInBounds(n, p1_rowStart + i, p1_colStart + j) && isInBounds(n, p2_rowStart + i, p2_colStart + j)){
                ans += pow((image[p1_rowStart + i][p1_colStart + j] - image[p2_rowStart + i][p2_colStart + j]), 2);
            }
        }
    }

    return sqrt(ans);
}

double computeWeight(double dist, double sigma, double z) 
{
    return (1 / z) * exp(dist / pow(sigma, 2));
}

} // namespace util

#endif // __UTILS_H__