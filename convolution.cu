
#include <cuda_runtime.h>
#include<iostream>
#include<time.h>
#include<random>

// nvcc convolution.cu -I /usr/local/include/opencv4/ -L /usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui

#define TILE_WIDTH 32
#define w_gauss (TILE_WIDTH + KER - 1)

__constant__ float kernel[KER*KER];

void gaussian_kernel(float *kernel, int radius, float sigma){
    float sum = 0.0;
    for (int i = -radius; i <= radius; ++i){
        for (int j = -radius; j <= radius; ++j){
            kernel[(i+radius)*KER + (j+radius)] = exp(-(i*i + j*j)/(2*sigma*sigma));
            sum += kernel[(i+radius)*KER + (j+radius)];
        }
    }
    for (int i = 0; i < KER*KER; ++i){
        kernel[i] /= sum;
    }
}

void edge_detection_kernel(float *kernel){
    kernel[0] = 0; kernel[1] = -1.0; kernel[2] = 0;
    kernel[3] = -1.0; kernel[4] = 5.0; kernel[5] = -1.0;
    kernel[6] = 0; kernel[7] = -1.0; kernel[8] = 0;
}

uint64_t nanos(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec*1000000000 + ts.tv_nsec;
}

__global__ void dumb_device_convolution(const uchar *img, const float *ker, uchar *out, const int H, const int W){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ker_r = KER/2;
    if((i < W-ker_r && j < H-ker_r) && (i >= ker_r && j >= ker_r)){
        for(int k = 0; k < KER; k++){
            for(int l = 0; l < KER; l++){
                out[i*W+j] += (uchar)img[(i-ker_r+k)*W+(j-ker_r+l)]*ker[k*KER+l];
            }
        }
    }
}

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

