#include <iostream>

using namespace std;
#include <stdint.h>
#include "cuda_runtime.h"
#include "cuda_test.h"

int main()
{
// ----- Cuda function test ----------
    
    int dev_count = 0;
    cudaGetDeviceCount( &dev_count);
    cudaDeviceProp dev_prop;
    for (int i=0; i<dev_count; i++)
    {
        cudaGetDeviceProperties( &dev_prop, i);
        cout << "i " << i << " max threads per block " << dev_prop.maxThreadsPerBlock 
        << " multiProcessorCount " << dev_prop.multiProcessorCount << " clockRate "
        << dev_prop.clockRate << " dev_prop.maxGridSize[0] " << dev_prop.maxGridSize[0]
        << " dev_prop.maxThreadsDim[0] " << dev_prop.maxThreadsDim[0] << endl;
        // decide if device has sufficient resources and capabilities
    }
    uchar mat_size = 9;
    uchar h_mat1[mat_size], h_mat2[mat_size], h_mat3[mat_size];
    uchar *d_mat1, *d_mat2, *d_mat3;
    uchar mat_mem_size = mat_size * sizeof(uchar);

    cudaMalloc((void**) &d_mat1 , mat_mem_size);
    cudaMalloc((void**) &d_mat2 , mat_mem_size);
    cudaMalloc((void**) &d_mat3 , mat_mem_size);

    for(int i=0; i<9; i++)
    {
        h_mat1[i] = h_mat2[i] = i; 
    }

    cudaMemcpy(d_mat1, h_mat1, mat_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, mat_mem_size, cudaMemcpyHostToDevice);

    mat_multiply<<<1,mat_size>>>(d_mat1, d_mat2, d_mat3);

    cudaMemcpy(h_mat3, d_mat3, mat_mem_size, cudaMemcpyDeviceToHost);

    for(int i=0; i<9; i++)
    {
        cout << (int)h_mat3[i] << endl; 
    }

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_mat3);
    return 0;
    // -----------------------------------
}
