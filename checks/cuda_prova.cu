#include <stdio.h>
#include <cuda.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

static void HandleError(cudaError_t err, const char *file, int line){
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int main() {
    cuda_hello<<<10,10>>>();
    cudaDeviceSynchronize();
    return 0;
}