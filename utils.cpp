#pragma once

#include <time.h>

uint64_t nanos(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec*1000000000 + ts.tv_nsec;
}

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

void edge_detection(float *kernel){
    for (int i = 0; i < KER*KER; ++i){
        kernel[i] = -1;
    }
    kernel[KER*KER/2] = 8;
}

void sharpen(float *kernel, int radius){
    for (int i = -radius; i <= radius; ++i){
        for (int j = -radius; j <= radius; ++j){
            kernel[(i+radius)*KER + (j+radius)] = -1;
        }
    }
    kernel[radius*KER + radius] = 5;
    kernel[0] = 0;
    kernel[KER-1] = 0;
    kernel[(KER-1)*KER] = 0;
    kernel[KER*KER-1] = 0;

}