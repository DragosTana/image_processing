#pragma once

#include <time.h>
#include <omp.h>

#define KER 5

uint64_t nanos () {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec*1000000000 + ts.tv_nsec;
}

void gaussian_kernel (float* kernel, float sigma) {

    float sum = 0.0;
    int center = KER / 2;

    for (int i = 0; i < KER; ++i) {
        for (int j = 0; j < KER; ++j) {
            int x = i - center;
            int y = j - center;
            kernel[i * KER + j] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[i * KER + j];
        }
    }

    for (int i = 0; i < KER; ++i) {
        for (int j = 0; j < KER; ++j) {
            kernel[i * KER + j] /= sum;
        }
    }
}
