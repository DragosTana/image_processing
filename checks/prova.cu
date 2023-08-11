#include <iostream>

// CUDA kernel
__global__ void gpuKernel(int* array, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        array[tid] = array[tid] * array[tid];
    }
}

// Regular C++ function
void cpuFunction() {
    std::cout << "This is a regular C++ function.\n";
}

int main() {
    const int size = 10;
    int hostArray[size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int* deviceArray;

    cudaMalloc(&deviceArray, size * sizeof(int));
    cudaMemcpy(deviceArray, hostArray, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    gpuKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceArray, size);

    cudaMemcpy(hostArray, deviceArray, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(deviceArray);

    for (int i = 0; i < size; i++) {
        std::cout << hostArray[i] << " ";
    }
    std::cout << std::endl;

    cpuFunction();

    return 0;
}
