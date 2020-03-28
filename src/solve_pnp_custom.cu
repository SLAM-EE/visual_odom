#include <iostream>

using namespace std;
#include <stdint.h>
#include "cuda_runtime.h"
#include "solve_pnp_custom.h"

int solve_pnp_custom(cv::Mat& _opoints, std::vector<cv::Point2f>& _ipoints,
                    cv::Mat& _cameraMatrix, cv::Mat& _distCoeffs,
                    cv::Mat& _rvec, cv::Mat& _tvec, bool useExtrinsicGuess,
                    int iterationsCount, float reprojectionError, double confidence,
                    cv::Mat& _inliers, int flags)
{

    int count = _ipoints.size();
    cv::Mat h_opoints, h_ipoints( count, 1, CV_32FC2, (void*)&_ipoints[0]);
    h_opoints = _opoints.clone();
    std::vector<float> h_param[6];

    cv::Mat h_rvec(3, 1, CV_32F, &h_param[0]), h_tvec(3, 1, CV_32F, &h_param[3]);
    cv::Mat h_param_mat(6, 1, CV_32F, &h_param[0]);

    /*cv::solvePnPRansac( _opoints, _ipoints, _cameraMatrix, _distCoeffs, _rvec, _tvec,
                          useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                          _inliers, flags );*/
    cv::solvePnP( h_opoints, h_ipoints, _cameraMatrix, _distCoeffs, _rvec, _tvec,
                      useExtrinsicGuess, flags );

    solve_pnp_cuda2<<<1,1>>>();
    
    /*CvLevMarq solver(6, count*2, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,iterationsCount,FLT_EPSILON), true);
    cv::Mat * J=0;
    cv::Mat err(count*2, 1, CV_32F);

    while(1)
    {
        const CvMat *__param = 0;
        cv::Mat projected_points(count, 2, CV_32F);
        cv::Mat projected_diff(count, 2, CV_32F);
        bool proceed = solver.update(__param, J, &err);
        cvCopy( __param, &h_param_mat );
        if(!proceed) break;
        cvProjectPoints2( h_opoints, &h_rvec, &h_tvec, &_cameraMatrix, &_distCoeffs,
                              projected_points, 0, 0, 0, 0, 0 );
        cvSub(&projected_diff, &h_ipoints, &projected_points);
        err = projected_diff.reshape(count*2, 1).clone();
    }

    _rvec = h_rvec.clone();
    _tvec = h_tvec.clone();*/

/*--------------------------------------------------------------------
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

    solve_pnp_cuda<<<1,mat_size>>>(d_mat1, d_mat2, d_mat3);

    cudaMemcpy(h_mat3, d_mat3, mat_mem_size, cudaMemcpyDeviceToHost);

    for(int i=0; i<9; i++)
    {
        cout << (int)h_mat3[i] << endl; 
    }

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_mat3);
//-------------------------------------------------------*/

    return 0;
}
