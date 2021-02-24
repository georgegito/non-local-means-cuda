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

    bool useGpu = false;
    int n = 64;
    int patchSize;
    float filterSigma;
    float patchSigma;

    if (argc == 1) {
        patchSize = 5;
        filterSigma = 0.06;
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

    std::vector<float> image = file::read("./data/in/noisy_house.txt", n, n, ',');
    std::cout << "Image read" << std::endl;
    std::vector<float> filteredImage;

/* -------------------------------------------------------------------------- */
/*                             cpu image filtering                            */
/* -------------------------------------------------------------------------- */

    if (!useGpu) {
        timer.start("CPU Filtering");
        filteredImage = filterImage(image.data(), n, patchSize, patchSigma, filterSigma);
        timer.stop();
    }

/* -------------------------------------------------------------------------- */
/*                             gpu image filtering                            */
/* -------------------------------------------------------------------------- */

    if (useGpu) {
        timer.start("GPU Filtering");
        filteredImage = cudaFilterImage(image.data(), n, patchSize, patchSigma, filterSigma);
        timer.stop();
    }

/* ---------------------------- print parameters ---------------------------- */

    prt::parameters(patchSize, filterSigma, patchSigma);

/* ---------------------------- compute residual ---------------------------- */

    std::vector<float> residual = util::computeResidual(image, filteredImage, n);

/* ------------------------------ file writing ------------------------------ */

    std::string outPath = file::write_images(filteredImage, residual, patchSize, filterSigma, patchSigma , n, n, useGpu);

/* ------------------------------- output test ------------------------------ */
    if (!useGpu) {
        test::out(  "./data/standard/standard_5_0.06_0.8.txt", outPath, n  );
        test::out(  "./data/standard/standard_res_5_0.06_0.8.txt", 
                    "./data/out/residual_5_0.060000_0.800000.txt", n  ); 
    }
    else {
        test::out(  "./data/standard/cuda_standard_5_0.06_0.8.txt", outPath, n  );
        test::out(  "./data/standard/cuda_standard_res_5_0.06_0.8.txt", 
                    "./data/out/cuda_residual_5_0.060000_0.800000.txt", n  ); 
    }

/* ----------------------- compute mean squared error ----------------------- */
    
    float meanSquaredError;
    
    if (!useGpu)
        meanSquaredError = test::computeMeanSquaredError(   "./data/standard/house.txt", 
                                                            "./data/out/filtered_image_5_0.060000_0.800000.txt", n   );
    else
        meanSquaredError = test::computeMeanSquaredError(   "./data/standard/house.txt", 
                                                            "./data/out/cuda_filtered_image_5_0.060000_0.800000.txt", n   );

    std::cout << "Mean squared error = " << meanSquaredError << std::endl << std::endl;

/* --------------------------------------------------------------------------- */

    std::cout << std::endl;
    return 0;
}