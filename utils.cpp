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

void edge_detection_kernel(float *kernel){
    kernel[0] = 0; kernel[1] = -1.0; kernel[2] = 0;
    kernel[3] = -1.0; kernel[4] = 5.0; kernel[5] = -1.0;
    kernel[6] = 0; kernel[7] = -1.0; kernel[8] = 0;
}

