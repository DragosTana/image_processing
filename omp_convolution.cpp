#include "utils.cpp"
#include "omp.h"

/*
Naive OpenMP convolution. The outermost loop is parallelized using a simple #pragma omp parallel for directive.
@param image: input matrix
@param ker: kernel
@param out: output matrix
@param H: height of input matrix
@param W: width of input matrix
@param threads: number of threads
*/
void naive_omp_convolution (const uchar *image, const float *ker, uchar *out, const int H, const int W) {   

    int ker_r = KER/2;
    int thread_num = omp_get_max_threads();
    int chunk = ceil(H / (float) thread_num);
    #pragma omp parallel for num_threads(thread_num) shared(image, ker, out, ker_r) , schedule(static, chunk)
    for(int i = ker_r; i < W - ker_r; i++){
        for(int j = ker_r; j < H - ker_r; j++){
            for(int k = 0; k < KER; k++){
                for(int l = 0; l < KER; l++){
                    out[i*W+j] += (uchar)image[(i-ker_r+k)*W+(j-ker_r+l)]*ker[k*KER+l];
                }
            }
            }
        }
}

/*
Naive OpenMP convolution. The outermost loop is parallelized using a simple #pragma omp parallel for directive.
The convolution operation is vectorized using #pragma omp simd directive.
@param image: input matrix
@param ker: kernel
@param out: output matrix
@param H: height of input matrix
@param W: width of input matrix
@param threads: number of threads
*/
void vec_omp_convolution (const uchar *image, const float *ker, uchar *out, const int H, const int W) {   

    int ker_r = KER/2;
    int thread_num = omp_get_max_threads();
    int chunk = ceil(H / (float) thread_num);
    #pragma omp parallel for num_threads(thread_num) shared(image, ker, out, ker_r) , schedule(static, chunk)
    for(int i = ker_r; i < W - ker_r; i++){
        for(int j = ker_r; j < H - ker_r; j++){

            #pragma omp simd
            for(int k = 0; k < KER; k++){
                for(int l = 0; l < KER; l++){
                    out[i*W+j] += (uchar)image[(i-ker_r+k)*W+(j-ker_r+l)]*ker[k*KER+l];
                }
            }
            }
        }
}

/*
Smarter OpenMP convolution. Every core has a private output matrix, which is then merged into the global output matrix.
@param image: input matrix
@param K: kernel
@param out: output matrix
@param H: height of input matrix
@param W: width of input matrix
@param threads: number of threads
*/
void smart_omp_convolution(const uchar *image, const float *ker, uchar *out, const int H, const int W) {
    int ker_r = KER / 2;
    int thread_num = omp_get_max_threads();
    int chunk = ceil(H / (float) thread_num);

    #pragma omp parallel num_threads(thread_num) shared(image, ker, out, ker_r, chunk)
    {
        int start_row = omp_get_thread_num() * chunk;
        int end_row = (omp_get_thread_num() == thread_num - 1) ? H : start_row + chunk;
        
        uchar *private_out = new uchar[W * chunk]();

        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < W; j++) {
                for (int k = 0; k < KER; k++) {
                    for (int l = 0; l < KER; l++) {
                        if (i >= ker_r && i < H - ker_r && j >= ker_r && j < W - ker_r) {
                            private_out[(i - start_row) * W + j] += (uchar) image[(i - ker_r + k) * W + (j - ker_r + l)] * ker[k * KER + l];
                        } else {
                            private_out[(i - start_row) * W + j] = image[i * W + j];
                        }
                    }
                }
            }
        }

            for (int i = start_row; i < end_row; i++) {
                for (int j = 0; j < W; j++) {
                    out[i * W + j] = private_out[(i - start_row) * W + j];
                }
            }  

    delete[] private_out;
    }
}

/*
Wrapper for OpenMP convolution
@param M: input matrix
@param kernel_h: kernel
@param threads: number of threads 
*/
cv::Mat omp_convolution (const cv::Mat &image, const float kernel_h[KER*KER]) {
    
    double start = omp_get_wtime();
    cv::Mat out(image.rows, image.cols, CV_8UC1);
    vec_omp_convolution(image.data, kernel_h, out.data, image.rows, image.cols);
    double end = omp_get_wtime();
    std::cout <<(end-start)<<std::endl;
    return out;
}
