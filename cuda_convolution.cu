
#include <cuda_runtime.h>
#include<iostream>
#include<random>

#include "utils.cpp"

#define TILE_WIDTH_X 32
#define TILE_WIDTH_Y 16
#define INPUT_TILE_X (TILE_WIDTH_X + KER - 1)
#define INPUT_TILE_Y (TILE_WIDTH_Y + KER - 1)

__constant__ float ker[KER * KER];

/*
Cuda implementation of 2D convolution using constant memory and shared memory
@param image: input image
@param outputImageData: output image
@param width: image width
@param height: image height
*/
__global__ void device_convolution(const uchar *image, uchar *out, const int width, const int height) {

    __shared__ float N_ds[INPUT_TILE_Y][INPUT_TILE_X];
    int ker_r = KER / 2;

    int dest = threadIdx.y * TILE_WIDTH_X + threadIdx.x;
    int dest_y = dest / INPUT_TILE_X;
    int dest_x = dest % INPUT_TILE_X;
    int src_x = blockIdx.x * TILE_WIDTH_X + dest_x - ker_r;
    int src_y = blockIdx.y * TILE_WIDTH_Y + dest_y - ker_r;

    if (src_y >= 0 && src_y < height && src_x >= 0 && src_x < width)
        N_ds[dest_y][dest_x] = image[(src_y * width + src_x)];
    else
        N_ds[dest_y][dest_x] = 0;
        
    dest = threadIdx.y * TILE_WIDTH_X + threadIdx.x + TILE_WIDTH_X * TILE_WIDTH_Y;
    dest_y = dest / INPUT_TILE_X;
    dest_x = dest % INPUT_TILE_X;
    src_y = blockIdx.y * TILE_WIDTH_Y + dest_y - ker_r;
    src_x = blockIdx.x * TILE_WIDTH_X + dest_x - ker_r;
    if (dest_y < INPUT_TILE_Y)
    {
        if (src_y >= 0 && src_y < height && src_x >= 0 && src_x < width)
            N_ds[dest_y][dest_x] = image[(src_y * width + src_x)];
        else
            N_ds[dest_y][dest_x] = 0;
    }
    __syncthreads();

    float temp = 0;
    int y, x;
    for (y = 0; y < KER; y++)
        for (x = 0; x < KER; x++)
            temp += (N_ds[threadIdx.y + y][threadIdx.x + x] * ker[y * KER + x]);
            y = blockIdx.y * TILE_WIDTH_Y + threadIdx.y;
            x = blockIdx.x * TILE_WIDTH_X + threadIdx.x;
    if (y < height && x < width)
        //printf("%f %d\n", temp, (int)((uchar)temp));
        out[(y * width + x)] = (uchar)temp;
}


__global__ void dumb_device_convolution (uchar *image, uchar *out, int width, int height) {

    int ker_r = KER / 2;

    if (blockIdx.x * blockDim.x + threadIdx.x < width && blockIdx.y * blockDim.y + threadIdx.y < height) {
        float accum = 0;
        for (int i = 0; i < KER; i++) {
            for (int j = 0; j < KER; j++) {
                int y = blockIdx.y * blockDim.y + threadIdx.y - ker_r + i;
                int x = blockIdx.x * blockDim.x + threadIdx.x - ker_r + j;
                if (y >= 0 && y < height && x >= 0 && x < width) {
                    accum += image[y * width + x] * ker[i * KER + j];
                }
            }
        }        
        out[(blockIdx.y * blockDim.y + threadIdx.y) * width + blockIdx.x * blockDim.x + threadIdx.x] = (uchar)accum;
    }
}
/*
Wrapper for device convolution
@param image: input image as cv::Mat object
@param kernel_h: kernel
*/
cv::Mat cuda_convolution (const cv::Mat& image, const float kernel_h[KER*KER]) {
    
    uchar *d_input, *d_output;
    //size_t pitch;


    cudaMalloc(&d_input, image.rows*image.cols*sizeof(uchar));
    //cudaMallocPitch(&d_input, &pitch, image.cols*sizeof(uchar), image.rows);
    cudaMalloc(&d_output, image.rows*image.cols*sizeof(uchar));

    double start = omp_get_wtime();
    cv::Mat output(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
    cudaMemcpy(d_input, image.data, image.rows*image.cols*sizeof(uchar), cudaMemcpyHostToDevice);
    //cudaMemcpy2D(d_input, pitch, image.data, pitch, image.cols * sizeof(uchar), image.rows, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(ker, kernel_h, KER*KER*sizeof(float));

    dim3 dimGrid(ceil(image.cols/(float)TILE_WIDTH_X), ceil(image.rows/(float)TILE_WIDTH_Y), 1);
    dim3 dimBlock(TILE_WIDTH_X, TILE_WIDTH_Y, 1);
    device_convolution<<<dimGrid, dimBlock>>>(d_input, d_output, image.cols, image.rows);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output.data, d_output, image.rows*image.cols*sizeof(uchar), cudaMemcpyDeviceToHost);
    double end = omp_get_wtime();
    
    std::cout << (end-start)<<std::endl;
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
