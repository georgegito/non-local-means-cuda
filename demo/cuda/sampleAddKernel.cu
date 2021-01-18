#include <math.h>
#include <stdio.h>

// Array access macros
#define INPUT(i,j) A[(i) + (j)*(m)]
#define OUTPUT(i,j) B[(i) + (j)*(m)]

__global__ void sampleAdd(double const * const A, double *B, int m, int n) {
  // Get pixel (x,y) in input
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i<m && j<n) {

    OUTPUT(i,j) = INPUT(i,j) + 1;
    
  }
}