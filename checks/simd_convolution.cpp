#include<iostream>
#include<time.h>
#include<string>
#include<immintrin.h>
#include<random>


/*
* 1. Implement a naive convolution algorithm
* 2. Implement a convolution algorithm using SIMD instructions
* 3. Compare the performance of the two algorithms
*/

void avx_convolution(const float *image, const float *kernel, float *output, int image_size, int kernel_size){
    //transform kernel in toeplitz matrix

    float toeplitz[image_size * image_size * image_size * image_size] __attribute__ ((aligned (32)));
    for(int i = 0; i < image_size * image_size; i++){
        for(int j = 0; j < image_size * image_size; j++){
            if(i < kernel_size && j < kernel_size){
                toeplitz[i * image_size * image_size + j] = kernel[i * kernel_size + j];
            }
            else{
                toeplitz[i * image_size * image_size  + j] = 0;
            }
        }
    }
    
    __m256 *toeplitz_m = (__m256*) toeplitz;
    __m256 *image_m = (__m256*) image;
    __m256 *output_m = (__m256*) output;

    for(int j = 0; j < image_size; j++){
        __m256 sum = _mm256_setzero_ps();
        for(int i = 0; i < image_size; i = i + 8){
            sum = _mm256_fmadd_ps(toeplitz_m[j * image_size + i], image_m[i], sum);
        }
        output_m[j] = sum;
        }
    }


void naive_convolution(const float *image, const float *kernel, float *output, int image_size, int kernel_size){
    int output_size = image_size - kernel_size + 1;
    for(int i = 0; i < output_size; i++){
        for(int j = 0; j < output_size; j++){
            output[i * output_size + j] = 0;
            for(int k = 0; k < kernel_size; k++){
                for(int l = 0; l < kernel_size; l++){
                    output[i * output_size + j] += image[(i + k) * image_size + (j + l)] * kernel[k * kernel_size + l];
                }
            }
        }
    }
}

uint64_t nano(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000000 + t.tv_nsec;
}

bool check_consistency(const float *A, const float *B, const int H, const int W){
    for(int i = 1; i < H; i++){
        for(int j = 1; j < W; j++){
            if(A[i*W+j] != B[i*W+j]){
                std::cout << "Error at " << i << " " << j << std::endl;
                std::cout << "navie: " << A[i*W+j] << " simd: " << B[i*W+j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

void filter_creation (float GKernel[], const int radius, const float sigma) {
    float r, s = 2.0 * sigma * sigma;
    float sum = 0.0;
    int i, j;
    for (i = -radius; i <= radius; i++){
        for (j = -radius; j <= radius; j++){
            r = sqrt(i*i + j*j);
            GKernel[(i+radius)*2*radius+(j+radius)] = (exp(-(r*r)/s))/(M_PI * s);
            sum += GKernel[(i+radius)*2*radius+(j+radius)];
        }
    }
    for (i = 0; i < 2*radius+1; ++i)
        for (j = 0; j < 2*radius+1; ++j)
            GKernel[i*2*radius+j] /= sum;
}

int main() {
    int image_size = 32;
    int kernel_size = 3;
    
    float image[image_size * image_size] __attribute__ ((aligned (32)));
    float kernel[kernel_size * kernel_size] __attribute__ ((aligned (32)));
    float output[image_size * image_size] __attribute__ ((aligned (32)));

    std::default_random_engine randomGenerator(time(0));
    std::uniform_real_distribution<float> diceroll(0.0f, 1.0f);
    for(size_t i = 0; i < image_size * image_size; i++){
        image[i] = diceroll(randomGenerator);
    }
    filter_creation(kernel, kernel_size/2, 1);

    uint64_t start = nano();
    naive_convolution(image, kernel, output, image_size, kernel_size);
    uint64_t end = nano();
    std::cout << "Naive time: " << (end - start) * 1e-3 << " microseconds" << std::endl;

    float *output_simd = (float*) _mm_malloc(image_size * image_size * sizeof(float), 32);
    start = nano();
    avx_convolution(image, kernel, output_simd, image_size, kernel_size);
    end = nano();
    std::cout << "SIMD time: " << (end - start) * 1e-3 << " microseconds" << std::endl;

    if(check_consistency(output, output_simd, image_size, image_size)){
        std::cout << "Consistent" << std::endl;
    }
    

    return 0;
}


