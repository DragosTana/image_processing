#include <string>
#include <iostream>
#include <vector>

std::vector<std::string> convolution(std::vector<std::string> matrix, std::vector<std::string> kernel, int N, int K){
    std::vector<std::string> output = std::vector<std::string>((N-K+1)*(N-K+1));

    for(int i = 0; i < N-K+1; i++){
        for(int j = 0; j < N-K+1; j++){
            for(int k = 0; k < K; k++){
                for(int l = 0; l < K; l++){
                    output[i*(N-K+1) + j] += matrix[(i+k)*N + (j+l)] + "*" + kernel[k*K + l] + " + ";
                }
            }
            output[i*(N-K+1) + j] = output[i*(N-K+1) + j].substr(0, output[i*(N-K+1) + j].size()-3);
        }
    }

    return output;
}

void print(std::vector<std::string> matrix, int N){
    for(int i = 0; i < N; i ++){
        for(int j = 0; j < N; j++){
            std::cout << matrix[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(){

    int N = 5;
    int K = 3;

    std::vector<std::string> matrix = std::vector<std::string>(N*N);
    std::vector<std::string> kernel = std::vector<std::string>(K*K);

    // populate matrix
    for(int i = 0; i < N; i ++){
        for(int j = 0; j < N; j++){
            matrix[i*N + j] = "m" + std::to_string(i) + std::to_string(j);
        }
    }

    // populate kernel
    for(int i = 0; i < K; i ++){
        for(int j = 0; j < K; j++){
            kernel[i*K + j] = "k" + std::to_string(i) + std::to_string(j);
        }
    }

    std::vector<std::string> output = convolution(matrix, kernel, N, K);
    print(output, N-K+1);
}