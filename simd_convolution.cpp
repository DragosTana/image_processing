#include "utils.cpp"
#include <immintrin.h>
/*
Simd convolution obtained by unrolling the innermost loop and using #pragma omp simd
@param M: input matrix
@param K: kernel
@param out: output matrix
@param H: height of input matrix
@param W: width of input matrix
*/
void simd_convolution(const uchar __restrict_arr *image, const float __restrict_arr *ker, uchar __restrict_arr *out, const int H, const int W){
    int ker_r = KER/2;
    
    for(int i = ker_r; i < W-ker_r; i++){
        #pragma omp simd
        for(int j = ker_r; j < H-ker_r; j++){
            float sum = 0.0f; // Accumulate the sum using float (since ker is float)
            for(int k = 0; k < KER; k++){
                for(int l = 0; l < KER; l++){
                    sum += (uchar)image[(i-ker_r+k)*W+(j-ker_r+l)] * ker[k*KER+l];
                }
            }
        }
    }
}

/*
Wrapper for simd convolution
@param M: input matrix as cv::Mat object
@param kernel_h: kernel as float array
*/
cv::Mat vec_convolution(const cv::Mat &M, const float kernel_h[KER*KER]){
    
    double start = omp_get_wtime();
    cv::Mat out(M.rows, M.cols, CV_8UC1);
    host_convolution(M.data, kernel_h, out.data, M.rows, M.cols);
    double end = omp_get_wtime();
    std::cout <<"time: "<< (end-start)*1e-3 <<std::endl;
    return out;
}
