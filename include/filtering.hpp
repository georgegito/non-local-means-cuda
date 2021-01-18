#ifndef __FILTERING_H__
#define __FILTERING_H__

#include <utils.hpp>

double filter(std::vector<std::vector<int>> image, int n, int patchSize, int p1_row, int p1_col, int p2_row, int p2_col) {
    return util::computeEuclideanDistance(image, n, patchSize, p1_row, p1_col, p2_row, p2_col);
}

#endif // __FILTERING_H__