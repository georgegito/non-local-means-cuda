#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <vector>
#include <cmath>

namespace util {

// std::vector<std::vector<double>> computeDistanceMatrix(std::vector<std::vector<int>> image, int n)
// {
//     std::vector<std::vector<double>> D(n, std::vector<double>(n));

//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             for (int k = 0; k < n; k++) {
//                 for (int l = 0; l < n; l++) {
//                     D[i][j] = pow
//                 }
//             }
//         }
//     }

//     return D;
// }

bool isInBounds(int n, int x, int y) 
{
    return x >=0 && x < n && y >= 0 && y < n;
}

double computeEuclideanDistance(std::vector<std::vector<int>> image, std::vector<double> _weights, int n, int patchSize, int p1_row, int p1_col, int p2_row, int p2_col) 
{
    int p1_rowStart = p1_row - patchSize / 2;   // TODO avoid multiple computations for p1
    int p1_colStart = p1_col - patchSize / 2;
    int p2_rowStart = p2_row - patchSize / 2;
    int p2_colStart = p2_col - patchSize / 2;
    double ans = 0;

    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            if (isInBounds(n, p1_rowStart + i, p1_colStart + j) && isInBounds(n, p2_rowStart + i, p2_colStart + j)){
                ans += _weights[i * patchSize + j] * pow((image[p1_rowStart + i][p1_colStart + j] - image[p2_rowStart + i][p2_colStart + j]), 2);
            }
        }
    }

    return sqrt(ans);
}

double computeWeight(double dist, double sigma) // compute weight without "/z(i)" division
{
    return exp(-dist / pow(sigma, 2));
}

std::vector<double> computeInsideWeights(int patchSize, double patchSigma)
{
    std::vector<double> _weights(patchSize * patchSize);
    int centralPixelRow = patchSize / 2;
    int centralPixelCol = centralPixelRow;
    double _dist;
    double _sumW = 0;

    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            _dist = sqrt(pow(centralPixelRow - i, 2) + pow(centralPixelCol - j, 2));
            _weights[i * patchSize + j] = computeWeight(_dist, patchSigma);
            _sumW += _weights[i * patchSize + j];
        }
    }

    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize;j++) {
            _weights[i * patchSize + j] = _weights[i * patchSize + j] / _sumW;
        }
    }

    return _weights;
}


} // namespace util

namespace prt {

void rowMajorVector(std::vector<double> vector, int n, int m)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << vector[i * m + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

}

#endif // __UTILS_H__