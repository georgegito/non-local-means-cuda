#include <iostream>
#include <vector>
#include <filtering.cuh>
#include <cudaFiltering.cuh>
#include <cstdlib>
#include <string> 

int main(int argc, char** argv)
{   
    std::cout << std::endl;
    util::Timer timer(true);

/* ------------------------------- parameters ------------------------------- */

    bool isCuda;
    int n = 64;
    int patchSize;
    float filterSigma;
    float patchSigma;

    if (argc == 1) {
        patchSize = 5;
        filterSigma = 0.04;
        patchSigma = 0.8;
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

    std::vector<float> image(n * n);
    image = file::read("./data/in/noisy_house.txt", n, n, ',');

    std::cout << "Image read" << std::endl;

/* -------------------------------------------------------------------------- */
/*                             cpu image filtering                            */
/* -------------------------------------------------------------------------- */

    // timer.start("CPU Filtering");

    // std::vector<float> filteredImage = filterImage(image.data(), n, patchSize, patchSigma, filterSigma);

    // timer.stop();

    // isCuda = false;

/* -------------------------------------------------------------------------- */
/*                             gpu image filtering                            */
/* -------------------------------------------------------------------------- */

    timer.start("GPU Filtering");

    std::vector<float> filteredImage = cudaFilterImage(image.data(), n, patchSize, patchSigma, filterSigma);

    timer.stop();

    isCuda = true;

/* ---------------------------- print parameters ---------------------------- */

    std::cout   << "Image filtered: "   << std::endl
                << "-Patch size "       << patchSize    << std::endl
                << "-Patch sigma "      << patchSigma   << std::endl
                << "-Filter Sigma "     << filterSigma  << std::endl  << std::endl;


/* --------------------------- calculate residual --------------------------- */

    std::vector<float> residual(n * n);
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

    file::write_images(filteredImage, residual, params, n, n, isCuda);

    std::cout << "Filtered image written" << std::endl << std::endl;
    std::cout << "Residual written" << std::endl;

/* --------------------------------------------------------------------------- */

    std::cout << std::endl;
    return 0;
}