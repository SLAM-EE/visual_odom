#ifndef SOLVE_PNP_CUSTOM_H_
#define SOLVE_PNP_CUSTOM_H_

#include <iostream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/calib3d/calib3d_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc_c.h"

using namespace std;
#include <stdint.h>
#include "cuda_runtime.h"

typedef unsigned char uchar; 

__global__ void solve_pnp_cuda(uchar *A, uchar *B, uchar *C)
{
	uchar index = threadIdx.x;
	C[index] = A[index] + B[index];

}

__global__ void solve_pnp_cuda2()
{
	uchar index = threadIdx.x;
	uchar test_calc;
	test_calc = index * index;

}


#endif 
