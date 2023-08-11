
#include<iostream>
#include<time.h>
#include<string>
#include<immintrin.h>
#include<random>

#define N 512
#define BLOCK 4
#define KERNEL_SIZE 3

uint64_t nanos(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

void intrinsics(){
    float A[N*N] __attribute__ ((aligned (32)));
    float B[N*N] __attribute__ ((aligned (32)));
    float C_naive[N*N] __attribute__ ((aligned (32)));
    float C_simd[N*N] __attribute__ ((aligned (32)));

    std::default_random_engine randomGenerator(time(0));
    std::uniform_real_distribution<float> diceroll(0.0f, 1.0f);
    for(size_t i = 0; i<N*N; i++)
    {
        A[i] = diceroll(randomGenerator);
        B[i] = diceroll(randomGenerator);
    }
    
    //naive dot product
    uint64_t start = nanos();
    for(size_t i = 0; i<N*N; i++)
    {
       C_naive[i] = A[i] * B[i];
    }
    uint64_t end = nanos();
    std::cout<<"Naive time: " << (end - start)*1e-3 << "microseconds" << std::endl;

    //SIMD dot product with 128 bit registers
    uint64_t start2 = nanos();
    for(size_t i = 0; i < N*N; i+=4)
    {
        __m128 a  = _mm_load_ps(&A[i]);
        __m128 b  = _mm_load_ps(&B[i]);
        __m128 c = _mm_mul_ps(a, b);
        _mm_store_ps(&C_simd[i], c);
        
    }
    u_int64_t end2 = nanos();
    std::cout<<"SIMD 128 time: " << (end2 - start2)*1e-3 << "microseconds" << std::endl;

    //SIMD dot product with 256 bit registers
    uint64_t start3 = nanos();
    for (size_t i = 0; i < N*N; i+=8)
    {
        __m256 a = _mm256_load_ps(&A[i]);
        __m256 b = _mm256_load_ps(&B[i]);
        __m256 c = _mm256_mul_ps(a, b);
        _mm256_store_ps(&C_simd[i], c);
    }
    uint64_t end3 = nanos();
    std::cout<<"SIMD 256 time: " << (end3 - start3)*1e-3 << " microseconds" << std::endl;

    //check correctness
    for(size_t i = 0; i<N*N; i++)
    {
        if(C_naive[i] != C_simd[i])
        {
            std::cout << "Error: SIMD dot product is incorrect" << std::endl;
            return;
        }
    }
}

void instrisics2(){
    float A[N*N] __attribute__ ((aligned (32)));
    float B[N*N] __attribute__ ((aligned (32)));
    float C_simd[N*N] __attribute__ ((aligned (32)));
    float C_naive[N*N] __attribute__ ((aligned (32)));

    std::default_random_engine randomGenerator(time(0));
    std::uniform_real_distribution<float> diceroll(0.0f, 1.0f);
    for(size_t i = 0; i<N*N; i++)
    {
        A[i] = diceroll(randomGenerator);
        B[i] = diceroll(randomGenerator);
    }

    __m256 *Am = (__m256*)A;
    __m256 *Bm = (__m256*)B;
    __m256 *Cm = (__m256*)C_simd;

    //naive dot product
    uint64_t start1 = nanos();
    for(size_t i = 0; i<N*N; i++)
    {
       C_naive[i] = A[i] * B[i];
    }
    uint64_t end1 = nanos();
    std::cout<<"Naive time: " << (end1 - start1)*1e-3 << "microseconds" << std::endl;

    //SIMD dot product with 256 bit registers
    uint64_t start = nanos();
    for (size_t i = 0; i < N*N/8; i++)
    {
        Cm[i] = _mm256_mul_ps(Am[i], Bm[i]);
    }
    uint64_t end = nanos();

    std::cout<<"SIMD 256 time: " << (end - start)*1e-3 << " microseconds" << std::endl;
}

void fast_convolution(){
    uint8_t M[N*N] __attribute__ ((aligned (8)));
    uint8_t K[KERNEL_SIZE*KERNEL_SIZE] __attribute__ ((aligned (8)));
    uint8_t C_simd[N*N] __attribute__ ((aligned (8)));

    std::default_random_engine randomGenerator(time(0));
    std::uniform_real_distribution<float> diceroll(0, 255);

    for(size_t i = 0; i<N*N; i++)
    {
        M[i] = (uint8_t)diceroll(randomGenerator);
    }

    for(size_t i = 0; i<KERNEL_SIZE*KERNEL_SIZE; i++)
    {
        K[i] = (uint8_t)diceroll(randomGenerator);
    }

    //convolution
    __m256 *Mm = (__m256*)M;
    __m256 *Km = (__m256*)K;
    __m256 *Cm = (__m256*)C_simd;

    for (size_t i = 0; i < KERNEL_SIZE*KERNEL_SIZE; i++)
    {
        printf("%d ", K[i]);
        std::cout<< &Km[i] << std::endl;
    }

}

int main(){
    instrinsics2();
}
