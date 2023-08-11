#include "utils.cpp"
#include "omp.h"


void omp_convolution(const uchar *M, const float *K, uchar *out, const int H, const int W){   

    int ker_r = KER/2;
    #pragma omp parallel for num_threads(16) 
    for(int i = 0; i < W; i++){
        for(int j = 0; j < H; j++){
            for(int k = 0; k < KER; k++){
                for(int l = 0; l < KER; l++){
                    out[i*W+j] += (uchar)M[(i-ker_r+k)*W+(j-ker_r+l)]*K[k*KER+l];
                }
            }
            }
        }
}

cv::Mat host_omp_convolution(const cv::Mat &M, const float kernel_h[KER*KER]){
    
    uint64_t start = nanos();
    cv::Mat out(M.rows, M.cols, CV_8UC1);
    omp_convolution(M.data, kernel_h, out.data, M.rows, M.cols);
    uint64_t end = nanos();
    std::cout << "GFLOPS  omp: " << FLOP / (float)(end-start)<< " time: "<< (end-start)*1e-3 <<std::endl;
    return out;
}
