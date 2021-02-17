#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <vector>
#include <cmath>

namespace util {

// compute all-to-all-pixel squared distance: (p1_val - p2_val)^2
std::vector<std::vector<double>> computeDistanceMatrix(std::vector<std::vector<int>> image, int n)
{
    std::vector<std::vector<double>> D(n * n, std::vector<double>(n * n));

    for (int i = 0; i < n * n; i++) {
        for (int j = 0; j < n * n; j++) {
            D[i][j] = pow(image[i / n][i % n] - image[j / n][j % n], 2);
        }
    }

    return D;
}

// pixel-to-pixel squared distance from distance matrix
double indexDistanceMatrix( std::vector<std::vector<double>> D, 
                            int n, 
                            int p1_row, 
                            int p1_col, 
                            int p2_row, 
                            int p2_col )
{
    int _row = n * p1_row + p1_col;
    int _col = n * p2_row + p2_col;

    return D[_row][_col];
}

bool isInBounds(int n, int x, int y) 
{
    return x >= 0 && x < n && y >= 0 && y < n;
}

// patch-to-patch euclidean distance
double computeEuclideanDistance( std::vector<std::vector<int>> image, 
                                 std::vector<std::vector<double>> _distances, 
                                 std::vector<double> _weights, 
                                 int n, 
                                 int patchSize, 
                                 int p1_row, 
                                 int p1_col, 
                                 int p2_row, 
                                 int p2_col ) 
{
    int p1_rowStart = p1_row - patchSize / 2;
    int p1_colStart = p1_col - patchSize / 2;
    int p2_rowStart = p2_row - patchSize / 2;
    int p2_colStart = p2_col - patchSize / 2;
    double ans = 0;

    for (int i = 0; i < patchSize; i++) {
        // TODO check for improvement
        for (int j = 0; j < patchSize; j++) {
            if (isInBounds(n, p1_rowStart + i, p1_colStart + j) && isInBounds(n, p2_rowStart + i, p2_colStart + j)){
                // ans += _weights[i * patchSize + j] * pow((image[p1_rowStart + i][p1_colStart + j] - image[p2_rowStart + i][p2_colStart + j]), 2);
                ans += _weights[i * patchSize + j] * indexDistanceMatrix(_distances, n, p1_rowStart + i, p1_colStart + j, p2_rowStart + i, p2_colStart + j);
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

void twoDimVector(std::vector<std::vector<double>> vector, int n, int m) {

    for (int i = 0; i < n * n; i++) {
        for (int j = 0; j < n * n; j++) {
            std::cout << vector[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

} // namespace prt

#endif // __UTILS_H__