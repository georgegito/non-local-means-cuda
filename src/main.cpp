#include <iostream>
#include <vector>
#include <filtering.hpp>
#include <cstdlib>
#include <string> 

int main(int argc, char** argv)
{   
    std::cout << std::endl;

/* ------------------------- house image parameters ------------------------- */

    int n = 64;
    int patchSize;
    double filterSigma;
    double patchSigma;
    if(argc == 1){
        patchSize = 5;
        filterSigma = 0.03;
        patchSigma = 0.7;
    }else if(argc == 4){
        patchSize = atoi(argv[1]);
        filterSigma = atof(argv[2]);
        patchSigma = atof(argv[3]);
    }else{
        return 1;
    }

    // std::cout << patchSize << "\t" << filterSigma << "\t" << patchSigma << std::endl;

/* ------------------------------ file reading ------------------------------ */

    std::vector<double> image(n * n);
    image = file::read("./noisy_house.txt", n, n);

    std::cout << "Read the image" << std::endl;

/* ----------------------------- image filtering ---------------------------- */

    std::vector<double> filteredImage = filterImage(image, n, patchSize, patchSigma, filterSigma);

    std::cout << "Filtered the image with " << std::endl
                << "Patch size " << patchSize << std::endl
                << "Patch sigma " << patchSigma << std::endl
                << "Filter Sigma " << filterSigma << std::endl << std::endl;

    std::vector<double> res(n *n);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            res[i * n + j] = filteredImage[i * n + j] - image[i * n + j];
        }
    }

    std::cout << "Calculated the residual" << std::endl;

/* ------------------------------ file writing ------------------------------ */

    std::string dataPath = "data/";
    std::string filteredPath = dataPath + "filtered_image_" + 
                                argv[1] + "_" + 
                                argv[2] + "_" + 
                                argv[3]; 
    file::write(filteredImage, filteredPath, n, n);


    std::string resPath = dataPath + "residual_" + 
                                argv[1] + "_" + 
                                argv[2] + "_" + 
                                argv[3]; 
    file::write(res, resPath, n, n);

    std::cout << "Wrote the filtered image and the residual" << std::endl;

/* -------------------------------------------------------------------------- */

    std::cout << std::endl;
    return 0;
}