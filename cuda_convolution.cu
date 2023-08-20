
#include <cuda_runtime.h>
#include<iostream>
#include<random>

#include "utils.cpp"

#define TILE_WIDTH 32
#define w_gauss (TILE_WIDTH + KER - 1)

__constant__ float kernel[KER*KER];

/*
Cuda implementation of convolution
@param InputImageData: input image
@param outputImageData: output image
@param width: image width
@param height: image height
*/
__global__ void smart_device_convolution(uchar *InputImageData, uchar *outputImageData, int width, int height)
{
    __shared__ uint8_t N_ds[w_gauss][w_gauss];

    int maskRadius = KER / 2;
    
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destY = dest / w_gauss;
    int destX = dest % w_gauss;
    int srcY = blockIdx.y * TILE_WIDTH + destY - maskRadius;
    int srcX = blockIdx.x * TILE_WIDTH + destX - maskRadius;
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = InputImageData[(srcY * width + srcX)];
    else
        N_ds[destY][destX] = 0;
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest / w_gauss;
        destX = dest % w_gauss;
        srcY = blockIdx.y * TILE_WIDTH + destY - maskRadius;
        srcX = blockIdx.x * TILE_WIDTH + destX - maskRadius;
    if (destY < w_gauss)
    {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = InputImageData[(srcY * width + srcX)];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();

    uchar accum = 0;
    int y, x;
    for (y = 0; y < KER; y++)
        for (x = 0; x < KER; x++)
            accum += (uchar)N_ds[threadIdx.y + y][threadIdx.x + x] * kernel[y * KER + x];
            y = blockIdx.y * TILE_WIDTH + threadIdx.y;
            x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (y < height && x < width)
        outputImageData[(y * width + x)] = accum;
    __syncthreads();
}

/*
Wrapper for device convolution
@param image: input image as cv::Mat object
@param kernel_h: kernel
*/
cv::Mat device_convolution(const cv::Mat& image, const float kernel_h[KER*KER]){
    
    uchar *d_input, *d_output;

    cudaMalloc(&d_input, image.rows*image.cols*sizeof(uchar));
    cudaMalloc(&d_output, image.rows*image.cols*sizeof(uchar));

    uint64_t start = nanos();
    cv::Mat output(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
    cudaMemcpy(d_input, image.data, image.rows*image.cols*sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel, kernel_h, KER*KER*sizeof(float));
    dim3 dimGrid(ceil(image.cols/(float)TILE_WIDTH), ceil(image.rows/(float)TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    smart_device_convolution<<<dimGrid, dimBlock>>>(d_input, d_output, image.cols, image.rows);
    cudaDeviceSynchronize();
    cudaMemcpy(output.data, d_output, image.rows*image.cols*sizeof(uchar), cudaMemcpyDeviceToHost);
    uint64_t end = nanos();
    std::cout << "GFLOPS device: " << FLOP / (float)(end-start)<< " time: "<< (end-start)*1e-3 <<std::endl;
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
