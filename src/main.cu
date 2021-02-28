#include <iostream>
#include <vector>
#include <filtering.cuh>
#include <cudaFilteringGlobalMem.cuh>
#include <cudaFilteringSharedMem.cuh>
#include <cstdlib>
#include <string> 

int main(int argc, char** argv)
{   
    std::cout << std::endl;
    util::Timer timer(true);

/* ------------------------------- parameters ------------------------------- */

    int n;
    int imageNum;
    int patchSize;
    float filterSigma;
    float patchSigma;
    bool useGpu;
    bool useSharedMem;
    std::string imagePath;

    if (argc == 1) {
        imageNum = 0;
        patchSize = 5;
        filterSigma = 0.06;
        patchSigma = 0.8;
        useGpu = false;
        useSharedMem = false;
    }
    else if(argc == 7) {
        imageNum = atoi(argv[1]);
        patchSize = atoi(argv[2]);
        filterSigma = atof(argv[3]);
        patchSigma = atof(argv[4]);
        useGpu = atoi(argv[5]);
        useSharedMem = atoi(argv[6]);
    }
    else {
        return 1;
    }

    if (imageNum == 0) {
        n = 64;
        imagePath = "./data/in/noisy_house.txt";
        std::cout << "Image: House" << std::endl << std::endl;
    }

    else if (imageNum == 1) {
        n = 128;
        imagePath = "./data/in/noisy_flower.txt";
        std::cout << "Image: Flower" << std::endl << std::endl;
    }

    else if (imageNum == 2) {
        n = 256;
        imagePath = "./data/in/noisy_lena.txt";
        std::cout << "Image: Lena" << std::endl << std::endl;
    }

    else
        return 1;
/* ------------------------------ file reading ------------------------------ */

    std::vector<float> image = file::read(imagePath, n, n, ',');
    std::cout << "Image read" << std::endl;
    std::vector<float> filteredImage;

/* -------------------------------------------------------------------------- */
/*                             cpu image filtering                            */
/* -------------------------------------------------------------------------- */

    if (!useGpu) {
        timer.start("CPU filtering");
        filteredImage = cpu::filterImage(image.data(), n, patchSize, patchSigma, filterSigma);
        timer.stop();
    }

/* -------------------------------------------------------------------------- */
/*                             gpu image filtering                            */
/* -------------------------------------------------------------------------- */

    if (useGpu) {
        if (!useSharedMem) {
            timer.start("GPU filtering (global memory)");
            filteredImage = gpuGlobalMem::filterImage(image.data(), n, patchSize, patchSigma, filterSigma);
            timer.stop();
        }
        else {
            timer.start("GPU filtering (shared memory)");
            filteredImage = gpuSharedMem::filterImage(image.data(), n, patchSize, patchSigma, filterSigma);
            timer.stop();
        }
    }

/* ---------------------------- print parameters ---------------------------- */

    prt::parameters(patchSize, filterSigma, patchSigma);

/* -------------------------- residual computation -------------------------- */

    std::vector<float> residual = util::computeResidual(image, filteredImage, n);

/* ------------------------------ file writing ------------------------------ */

    std::string outPath = file::write_images(filteredImage, residual, patchSize, filterSigma, patchSigma , n, n, useGpu);

/* --------------------------------------------------------------------------- */

    std::cout << std::endl;
    return 0;
}