#ifndef CUDA_TEST_H_
#define CUDA_TEST_H_

#include <iostream>

using namespace std;
#include <stdint.h>
#include "cuda_runtime.h"

typedef unsigned char uchar; 

__global__ void mat_multiply(uchar *A, uchar *B, uchar *C)
{
	uchar index = threadIdx.x;
	C[index] = A[index] + B[index];
}


#endif 
