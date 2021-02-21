#include <iostream>
#include <vector>
#include <filtering.hpp>
#include <cstdlib>
#include <string> 

/* -------------------------------------------------------------------------- */
/*                     non-local-means cpu implementation                     */
/* -------------------------------------------------------------------------- */

int main(int argc, char** argv)
{   
    std::cout << std::endl;
    util::Timer timer(true);

/* ------------------------------- parameters ------------------------------- */

    int n = 64;
    int patchSize;
    double filterSigma;
    double patchSigma;

    if (argc == 1) {
        patchSize = 5;
        filterSigma = 0.03;
        patchSigma = 0.7;
    }
    else if(argc == 4) {
        patchSize = atoi(argv[1]);
        filterSigma = atof(argv[2]);
        patchSigma = atof(argv[3]);
    }
    else {
        return 1;
    }

/* ------------------------------ file reading ------------------------------ */

    std::vector<double> image(n * n);
    image = file::read("./data/in/noisy_house.txt", n, n);

    std::cout << "Image read" << std::endl;

/* ----------------------------- image filtering ---------------------------- */

    timer.start("Filtering");

    std::vector<double> filteredImage = filterImage(image, n, patchSize, patchSigma, filterSigma);

    timer.stop();

    std::cout   << "Image filtered: "   << std::endl
                << "-Patch size "                << patchSize    << std::endl
                << "-Patch sigma "               << patchSigma   << std::endl
                << "-Filter Sigma "              << filterSigma  << std::endl  << std::endl;

    std::vector<double> residual(n *n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            residual[i * n + j] = image[i * n + j] - filteredImage[i * n + j];
        }
    }

    std::cout << "Residual calculated" << std::endl << std::endl;

/* ------------------------------ file writing ------------------------------ */

    std::string params = std::to_string(patchSize)   + "_" + 
                         std::to_string(filterSigma) + "_" + 
                         std::to_string(patchSigma);

    file::write_images(filteredImage, residual, params, n, n);

    std::cout << "Filtered image written" << std::endl << std::endl;
    std::cout << "Residual written" << std::endl;

/* -------------------------------------------------------------------------- */

    std::cout << std::endl;
    return 0;
}