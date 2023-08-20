#include <opencv2/core.hpp>      
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#define KER 5
#define N 512
#define FLOP N*N*KER*KER*(KER-1)

#include "cuda_convolution.cu"
#include "convolution.cpp"
#include "omp_convolution.cpp"

// nvcc main.cu -I /usr/local/include/opencv4/ -L /usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lgomp -o main -O2 -Xcompiler -fopenmp
int main(){
    // cv::Mat img(N, N, CV_8UC1);
    // cv::randu(img, 0, 255);

    cv::Mat img = cv::imread("images/lenna_gray_512.jpg", cv::IMREAD_GRAYSCALE);
    
    float kernel_h[KER*KER];
    edge_detection(kernel_h);


    std::cout << "FLOP: " << FLOP << std::endl;
    cv::Mat out_3 = host_omp_convolution(img, kernel_h, 16);

    cv::imshow("Original", img);
    //cv::imshow("convolution_device", out_1);
    cv::imshow("convolution_host", out_3);
    cv::waitKey(0);

}