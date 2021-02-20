#include <iostream>
#include <vector>
#include <filtering.hpp>

int main()
{   
    std::cout << std::endl;

/* ------------------------- house image parameters ------------------------- */

    int n = 64;
    int patchSize = 5;
    double filterSigma = 0.06;
    double patchSigma = 0.8;
    // double filterSigma = 0.02;
    // double patchSigma = 5/3;

/* ------------------------------ file reading ------------------------------ */

    std::vector<double> image(n * n);
    image = file::read("./noisy_house.txt", n, n);

    // prt::rowMajorVector(image, n, n);

/* ----------------------------- image filtering ---------------------------- */

    std::vector<double> filteredImage = filterImage(image, n, patchSize, patchSigma, filterSigma);
    
    // std::cout << "filtered image:\n\n";
    // prt::rowMajorVector(filteredImage, n, n);

/* ------------------------------ file writing ------------------------------ */

    file::write(filteredImage, "filtered_image", n, n);

/* -------------------------------------------------------------------------- */

    std::cout << std::endl;
    return 0;
}