#ifndef __UTILS_CUH__
#define __UTILS_CUH__

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iterator>
#include <fstream>
#include <sstream>
#include <chrono>

namespace util {

class Timer {
  public:
    Timer(bool print) : print(print) {}

    void start(std::string operation_desc)
    {
        _operation_desc = operation_desc;
        t1              = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        t2       = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        if (print)
            std::cout << "\n" << _operation_desc << " time: " << duration / 1e3 << "ms\n" << std::endl;
    }

  private:
    double duration;
    bool print;
    std::string _operation_desc;
    std::chrono::high_resolution_clock::time_point t1, t2;
};

// compute all-to-all-pixel squared distance: (p1_val - p2_val)^2
std::vector<double> computeDistanceMatrix(std::vector<double> image, int n)
{
    std::vector<double> D(pow(n, 4));

    for (int i = 0; i < n * n; i++) {
        for (int j = 0; j < n * n; j++) {
            // D[i][j] = pow(image[i / n][i % n] - image[j / n][j % n], 2);
            D[i * n * n + j] = pow(image[(i / n) * n + i % n] - image[(j / n) * n + j % n], 2);
        }
    }

    return D;
}

// pixel-to-pixel squared distance from distance matrix
double indexDistanceMatrix( std::vector<double> D, 
                            int n, 
                            int p1_row, 
                            int p1_col, 
                            int p2_row, 
                            int p2_col )
{
    int _row = n * p1_row + p1_col;
    int _col = n * p2_row + p2_col;

    return D[_row * n * n + _col];
}

__host__ __device__ bool isInBounds(int n, int x, int y) 
{
    return x >= 0 && x < n && y >= 0 && y < n;
}

// patch-to-patch euclidean distance
__host__ __device__ double computePatchDistance( double* image, 
                             double* _weights, 
                             int n, 
                             int patchSize, 
                             int p1_rowStart, 
                             int p1_colStart, 
                             int p2_rowStart, 
                             int p2_colStart ) 
{
    double ans = 0;

    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            if (isInBounds(n, p1_rowStart + i, p1_colStart + j) && isInBounds(n, p2_rowStart + i, p2_colStart + j)) {
                ans += _weights[i * patchSize + j] * pow((image[(p1_rowStart + i) * n + p1_colStart + j] - image[(p2_rowStart + i) * n + p2_colStart + j]), 2);
                // ans += _weights[i * patchSize + j] * indexDistanceMatrix(_distances, n, p1_rowStart + i, p1_colStart + j, p2_rowStart + i, p2_colStart + j);
            }
        }
    }

    return ans;
}

__host__ __device__ double computeWeight(double dist, double sigma) // compute weight without "/z(i)" division
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
            _dist = pow(centralPixelRow - i, 2) + pow(centralPixelCol - j, 2);
            // _weights[i * patchSize + j] = computeWeight(_dist, patchSigma);
            _weights[i * patchSize + j] = exp(-_dist / (2 * pow(patchSigma, 2)));
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

} // namespace prt

namespace file {

std::vector<double> read(std::string filePath, int n, int m, char delim) 
{
    std::vector<double> image(n * m);
    std::ifstream myfile(filePath);
    std::ifstream input(filePath);
    std::string s;

    for (int i = 0; i < n; i++) {
        std::getline(input, s);
        std::istringstream iss(s);
        std::string num;
        int j = 0;
        while (std::getline(iss, num, delim)) {
            image[i * m + j++] = std::stof(num);
        }
    }

    return image;
}

void write(std::vector<double> image, std::string fileName, int rowNum, int colNum)
{
    std::vector<std::string> out;

    for (int i = 0; i < rowNum; i++) {
        for (int j = 0; j < colNum; j++) {
            out.push_back(std::to_string(image[i * colNum + j]) + " ");
        }
        out.push_back("\n");
    }

    std::ofstream output_file("./data/out/" + fileName + ".txt");
    std::ostream_iterator<std::string> output_iterator(output_file, "");
    std::copy(out.begin(), out.end(), output_iterator);
}

void write_images(std::vector<double> filteredImage, std::vector<double> residual, std::string params, int rowNum, int colNum, bool isCuda)
{                            
    std::string filteredName = "filtered_image_" + params;       
    if (isCuda) {
        filteredName = "cuda_" + filteredName;
    }
    file::write(filteredImage, filteredName, rowNum, colNum);

    std::string resName = "residual_" + params;
    if (isCuda) {
        resName = "cuda_" + resName;
    }
    file::write(residual, resName, rowNum, colNum);
}

} // namespace file

namespace test {

bool mat(std::vector<double> mat_1, std::vector<double> mat_2, int n)
{
    for (int i = 0; i< n; i++){
        for (int j=0; j < n; j++){
            if (mat_1[i*n + j] != mat_2[i*n + j]){
                return false;
            }
        }
    }
    return true;
}

void out(std::vector<double> out, int n, int m)
{
    std::vector<double> expectedOut = file::read("./data/out/expected_out.txt", n, m, ' ');
    
    // prt::rowMajorVector(expectedOut, n, m);

    int flag = 0;
    
    for (int i = 0; i < n * m; i++) {
        if (out[i] - std::fmod(out[i], 0.01) != expectedOut[i] - std::fmod(expectedOut[i], 0.01))
            flag = 1;
    }

    if (flag == 0) 
        std::cout << "Test passed" << std::endl << std::endl;
    else
        std::cout << "Test failed" << std::endl << std::endl;

}

} // namespace test

#endif // __UTILS_CUH__