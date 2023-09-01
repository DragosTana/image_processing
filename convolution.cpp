#include "utils.cpp"

/*
Naive convolution implementation
@param M: input matrix
@param K: kernel
@param out: output matrix
@param H: height of input matrix
@param W: width of input matrix
*/
void host_convolution(const uchar *image, const float  *ker, uchar *out, const int H, const int W){   
    int ker_r = KER/2;
    for(int i = ker_r; i < W-ker_r; i++){
        for(int j = ker_r; j < H-ker_r; j++){
            float temp = 0;
            for(int k = 0; k < KER; k++){
                for(int l = 0; l < KER; l++){
                    temp += ker[k*KER+l]*image[(j+k-ker_r)*W+(i+l-ker_r)];
                }
            }
            out[j*W+i] = temp;
            }
        }
}

/*
Just for fun
@param M: input matrix
@param K: kernel
@param out: output matrix
@param H: height of input matrix
@param W: width of input matrix
*/
void double_pixel (const uchar *image, const float  *ker, uchar *out, const int H, const int W){   

    for (int i = 0; i < H; i+=2){
        for (int j = 0; j < W; j+=2){
            out[i*W+j] = (uchar)2*image[i*W+j];
            
        }
    }

}

/*
Wrapper for host convolution
@param M: input matrix as cv::Mat object
@param kernel_h: kernel as float array
*/
cv::Mat seq_convolution(const cv::Mat &M, const float kernel_h[KER*KER]){
    
    double start = omp_get_wtime();
    cv::Mat out(M.rows, M.cols, CV_8UC1);
    double_pixel(M.data, kernel_h, out.data, M.rows, M.cols);
    double end = omp_get_wtime();
    std::cout <<(end-start)<<std::endl;
    return out;
}
