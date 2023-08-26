#include <opencv2/core.hpp>      
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>

#include "cuda_convolution.cu"
#include "convolution.cpp"
#include "omp_convolution.cpp"
#include "simd_convolution.cpp"

// Compile with:
// nvcc main.cu -I /usr/local/include/opencv4/ -L /usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lgomp -o main -O2 -Xcompiler -fopenmp

#define TEST 1

int release(int argc, const char *argv[]){
    std::string file_name = argv[1];
    std::string kernel = argv[2];
    std::string algorithm = argv[3];
    cv::Mat img = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
    if (img.empty()){
        std::cout << "Could not read the image: " << file_name << std::endl;
        return 1;
    }
    float kernel_h[KER*KER];
    if (kernel == "blur") {
        gaussian_kernel(kernel_h, 1);
    }
    else {
        std::cout << "Invalid kernel" << std::endl;
        return 1;
    }
    if (algorithm == "cuda") {
        cv::Mat out = cuda_convolution(img, kernel_h);
        //cv::imshow("Original", img);
        //cv::imshow("Convolution", out);
        //cv::waitKey(0);
    }
    else if (algorithm == "seq") {
        cv::Mat out = seq_convolution(img, kernel_h);
        //cv::imshow("Original", img);
        //cv::imshow("Convolution", out);
        //cv::waitKey(0);
    }
    else if (algorithm == "omp") {
        cv::Mat out = host_omp_convolution(img, kernel_h);
        cv::imshow("Original", img);
        cv::imshow("Convolution", out);
        cv::waitKey(0);
    }
    else if (algorithm == "simd") {
        cv::Mat out = vec_convolution(img, kernel_h);
        //cv::imshow("Original", img);
        //cv::imshow("Convolution", out);
        //cv::waitKey(0);
    }
    else {
        std::cout << "Invalid algorithm" << std::endl;
        return 1;
    }
    return 0;
}

int test(int argc, const char *argv[]){
    int N = atoi(argv[1]);
    std::string kernel = argv[2];
    std::string algorithm = argv[3];
    cv::Mat img = cv::Mat(N, N, CV_8UC1);
    cv::randu(img, cv::Scalar(0), cv::Scalar(255));
    float kernel_h[KER*KER];
    if (kernel == "blur") {
        gaussian_kernel(kernel_h, 1);
    }
    else {
        std::cout << "Invalid kernel" << std::endl;
        return 1;
    }
    if (algorithm == "cuda") {
        cv::Mat out = cuda_convolution(img, kernel_h);
    }
    else if (algorithm == "seq") {
        cv::Mat out = seq_convolution(img, kernel_h);
    }
    else if (algorithm == "omp") {
        cv::Mat out = host_omp_convolution(img, kernel_h);
    }
    else if (algorithm == "simd") {
        cv::Mat out = vec_convolution(img, kernel_h);
    }
    else {
        std::cout << "Invalid algorithm" << std::endl;
        return 1;
    }
    return 0;
}

int main(int argc, const char *argv[]){
#if TEST
    test(argc, argv);
#else
    release(argc, argv);
#endif
}