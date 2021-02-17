#include <iostream>
#include <vector>
#include <filtering.hpp>

int main()
{   
    std::cout << std::endl;

/* ---------------------------- data declaration ---------------------------- */

    // std::vector<double> image {
    //     0.541176470588235, 0.466666666666667, 0.439215686274510, 0.450980392156863, 0.458823529411765, 0.435294117647059, 0.458823529411765,
    //     0.600000000000000, 0.517647058823530, 0.486274509803922, 0.478431372549020, 0.494117647058824, 0.466666666666667, 0.458823529411765,
    //     0.541176470588235, 0.458823529411765, 0.447058823529412, 0.458823529411765, 0.474509803921569, 0.482352941176471, 0.490196078431373,
    //     0.498039215686275, 0.462745098039216, 0.454901960784314, 0.470588235294118, 0.462745098039216, 0.447058823529412, 0.450980392156863,
    //     0.474509803921569, 0.474509803921569, 0.482352941176471, 0.545098039215686, 0.517647058823530, 0.501960784313726, 0.501960784313726,
    //     0.501960784313726, 0.447058823529412, 0.450980392156863, 0.521568627450980, 0.498039215686275, 0.474509803921569, 0.478431372549020,
    //     0.568627450980392, 0.494117647058824, 0.486274509803922, 0.498039215686275, 0.486274509803922, 0.470588235294118, 0.458823529411765
    // };

    // std::vector<std::vector<int>> image {
    //     {1,     3,      4,      5,      1},
    //     {3,     5,      2,      8,      5},
    //     {4,     4,      2,      6,      1},
    //     {0,     8,      7,      4,      1},
    //     {0,     9,      0,      2,      3}
    // };

    std::vector<std::vector<int>> image {
        {1,     3,      4}, // (0,0) -> 0 * 3 + 0 = 0 col
        {3,     5,      2}, // (1,2) -> 1 * 3 + 2 = 5 row
        {4,     4,      2}
    };


    int n = image.size();
    // int patchSize = 3;
    int patchSize = 1;
    double filterSigma = 1;
    double patchSigma = 1.2;

/* ------------------------- euclidean distance test ------------------------ */

    // std::cout << "dist = " << util::computeEuclideanDistance(image,n, patchSize, 0, 0, 4, 0);
    // std::cout << std::endl;

/* ------------------------------- weight test ------------------------------ */

    // std::cout << "w = " << util::computeWeight(0, 1);
    // std::cout << std::endl;

/* ----------------------------- filtering test ----------------------------- */

    // int row = 0;
    // int col = 0;
    // std::cout << "* filtering pixel (" << row << ", " << col << ") *\n\n";
    // double res = filterPixel(image, n, patchSize, row, col, filterSigma);
    // std::cout << "initial pixel value = " << image[row][col] << " -> " << "filtered pixel value = " << res << std::endl;
    // std::cout << std::endl;

/* --------------------------- inside weights test -------------------------- */

    // std::vector<double> _weights = util::computeInsideWeights(3, patchSigma);
    // prt::rowMajorVector(_weights, 3, 3);

    // double _sum = 0;
    // for (auto v:_weights) {
    //     _sum += v;
    // }
    // std::cout << "_sumW = " << _sum << std::endl;

/* --------------- filtering with inside-gaussian-kernel test --------------- */

    // std::vector<double> _weights = util::computeInsideWeights(patchSize, patchSigma);

    // int row = 1;
    // int col = 1;

    // std::cout << "* filtering pixel (" << row << ", " << col << ") *\n\n";
    // double res = filterPixel(image, _weights, n, patchSize, row, col, filterSigma);
    // std::cout << "initial pixel value = " << image[row][col] << " -> " << "filtered pixel value = " << res << std::endl;
    // std::cout << std::endl;

/* -------------------------- image filtering test -------------------------- */

    // std::vector<double> filteredImage(n * n);
    // std::vector<double> _weights = util::computeInsideWeights(patchSize, patchSigma);

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         filteredImage[i * n + j] = filterPixel(image, _weights, n, patchSize, i, j, filterSigma);
    //     }
    // }

    // std::cout << "filtered image:\n\n";
    // prt::rowMajorVector(filteredImage, n, n);

/* --------------------------------- random --------------------------------- */

    // std::vector<double> vector{0, 1, 2, 3, 4, 5, 6, 7, 8};
    // prt::rowMajorVector(vector, 3, 3);

    std::vector<std::vector<double>> D = util::computeDistanceMatrix(image, n);
    prt::twoDimVector(D, n, n);
    std::cout << util::indexDistanceMatrix(D, n, 2, 0, 2, 2) << std::endl;
    
/* -------------------------------------------------------------------------- */

    std::cout << std::endl;
    return 0;
}

