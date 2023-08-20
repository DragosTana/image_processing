#include "utils.cpp"
#include "omp.h"

/*
OpenMP convolution
@param M: input matrix
@param K: kernel
@param out: output matrix
@param H: height of input matrix
@param W: width of input matrix
@param threads: number of threads
*/
void omp_convolution (const uchar *M, const float *K, uchar *out, const int H, const int W, const int threads) {   

    int ker_r = KER/2;
    #pragma omp parallel for num_threads(threads) 
    for(int i = ker_r; i < W - ker_r; i++){
        for(int j = ker_r; j < H - ker_r; j++){
            for(int k = 0; k < KER; k++){
                for(int l = 0; l < KER; l++){
                    out[i*W+j] += (uchar)M[(i-ker_r+k)*W+(j-ker_r+l)]*K[k*KER+l];
                }
            }
            }
        }
}

/*
Smart OpenMP convolution
@param M: input matrix
@param K: kernel
@param out: output matrix
@param H: height of input matrix
@param W: width of input matrix
@param threads: number of threads
*/
void smart_omp_convolution (const uchar *M, const float *K, uchar *out, const int H, const int W, const int threads) {  
    
    int ker_r = KER/2;
    int chunk = ceil(H/(float)threads);
    //std::cout << "chunk: " << chunk << std::endl;
    uchar private_out[W * H/threads];

    #pragma omp parallel num_threads(threads) shared(M, K, out, ker_r) private(private_out)
    {   
        #pragma omp for schedule(static, chunk) 
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W;  j++) {
                //if (omp_get_thread_num() == 1) { std::cout << "i, j: " << i << ", " << j << std::endl;}
                for (int k = 0; k < KER; k++) {
                    for (int l = 0; l < KER; l++) {
                        if (i >= ker_r && i < H - ker_r && j >= ker_r && j < W - ker_r){
                            private_out[(i - omp_get_thread_num()*chunk)*W + j] += (uchar)M[(i-ker_r+k)*W+(j-ker_r+l)]*K[k*KER+l];
                        }                    
                        else {
                            private_out[(i - omp_get_thread_num()*chunk)*W + j] = M[i*W+j];
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            //std::cout << "thread after: " << omp_get_thread_num() << std::endl;
            for (int i = 0; i < chunk; i++) {
                for (int j = 0; j < W; j++) {
                    out[(i + omp_get_thread_num()*chunk)*W + j] = private_out[i*W+j];
                }
            }
        }
    }
}

/*
Wrapper for OpenMP convolution
@param M: input matrix
@param kernel_h: kernel
@param threads: number of threads 
*/
cv::Mat host_omp_convolution (const cv::Mat &M, const float kernel_h[KER*KER], const int threads) {
    
    uint64_t start = nanos();
    cv::Mat out(M.rows, M.cols, CV_8UC1);
    omp_convolution(M.data, kernel_h, out.data, M.rows, M.cols, threads);
    uint64_t end = nanos();
    std::cout << "GFLOPS  omp: " << FLOP / (float)(end-start)<< " time: "<< (end-start)*1e-3 <<std::endl;
    return out;
}
