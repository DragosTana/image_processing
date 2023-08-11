#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <time.h>


// scrip to check if opencv and cuda compiles and runs correctly when used together
// compile with:
// nvcc cuda_opencv_prova.cu -I /usr/local/include/opencv4/ -L /usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs

uint64_t nanos() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;}


__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1, 1>>>();
    cudaDeviceSynchronize();

    int size[] = {10, 10};
    std::cout << std::endl;
    cv::Mat image(2, size, CV_8UC1);
    cv::randu(image, 0, 256);
    std::cout << "matrix data: "<< std::endl << cv::format(image, cv::Formatter::FMT_PYTHON) << std::endl;

    int data[size[0]][size[1]];

    for (int i = 0; i < size[0]; i++){
        for(int j = 0; j < size[1]; j++){
            data[i][j] = image.at<uchar>(i, j);
        }
    } 

    std::cout << std::endl;

    for (int i = 0; i < size[0]; i++){
        for(int j = 0; j < size[1]; j++){
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    size_t num_elements = size[0] * size[1];
    size_t size_in_bytes = num_elements * sizeof(int);

    int* gpu_data;

    cudaError_t cudaStatus;
    
    std::cout << std::endl;
    cudaStatus = cudaMalloc((void**)&gpu_data, size_in_bytes);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    else {
        std::cout << "cudaMalloc success!" << std::endl;
    }

    std::cout << std::endl;
    cudaStatus = cudaMemcpy(gpu_data, data, size_in_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy (Host to Device) failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    else {
        std::cout << "cudaMemcpy (Host to Device) success!" << std::endl;
    }

    int* cpuData = new int[num_elements];
    cudaStatus = cudaMemcpy(cpuData, gpu_data, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy (Device to Host) failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Data stored on GPU: " << std::endl;
    for (size_t i = 0; i < size[0]; i++){
        for(size_t j = 0; j < size[1]; j++){
            std::cout << cpuData[i * size[1] + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] cpuData;
    return 0;
}