#include <iostream>
#include <vector>
#include <filtering.hpp>

int main()
{   
    std::cout << std::endl;

/* ------------------------- parameters declaration ------------------------- */

    // // int n = image.size();
    // int n = 5;
    // int patchSize = 1;                      // patchSize -> always odd number
    // // int patchSize = 1;
    // double filterSigma = 1;
    // double patchSigma = 1.2;

/* ---------------------------- house parameters ---------------------------- */

    int n = 64;
    int patchSize = 5;
    double filterSigma = 0.02;
    double patchSigma = 5/3;

/* ---------------------------- file reading test --------------------------- */

    std::vector<double> image(n * n);
    image = file::read("./noisy_house.txt", n, n);
    // image = file::read("./house.txt", n, n);

    // prt::rowMajorVector(image, n, n);

/* ---------------------------- data declaration ---------------------------- */

    // std::vector<double> image {
    //     1,     3,      4,      5,      1,
    //     3,     5,      2,      8,      5,
    //     4,     4,      2,      6,      1,
    //     0,     8,      7,      4,      1,
    //     0,     9,      0,      2,      3
    // };                                      // image -> always squared

    // std::vector<std::vector<int>> image {
    //     {1,     3,      4},
    //     {3,     5,      2},
    //     {4,     4,      2}
    // };

/* -------------------------- image filtering test -------------------------- */

    std::vector<double> filteredImage = filterImage(image, n, patchSize, patchSigma, filterSigma);
    
    std::cout << "filtered image:\n\n";
    // prt::rowMajorVector(filteredImage, n, n);

/* ---------------------------- file writing test --------------------------- */

    file::write(filteredImage, "filtered_image", n, n);

/* -------------------------------------------------------------------------- */

    std::cout << std::endl;
    return 0;
}