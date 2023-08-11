#include <opencv2/core.hpp>      
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#define KER 3
#include "convolution.cu"

// nvcc main.cu -I /usr/local/include/opencv4/ -L /usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui

int main(){
    cv::Mat img = cv::imread("images/lenna_gray_512.jpg", cv::IMREAD_GRAYSCALE);
    
    float kernel_h[KER*KER];
    gaussian_kernel(kernel_h, KER/2, 1.0);

    cv::Mat out_1 = device_convolution(img, kernel_h);

    cv::imshow("Original", img);
    cv::imshow("convolution", out_1);
    cv::waitKey(0);

}