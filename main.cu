#include <opencv2/core.hpp>      
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#define KER 3
#include "convolution.cu"
#include "convolution.cpp"

// nvcc main.cu -I /usr/local/include/opencv4/ -L /usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui

int main(){
    cv::Mat img = cv::imread("lenna_gray_512.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat out_1 = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    cv::Mat out_2 = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

    float kernel[KER*KER];
    edge_detection_kernel(kernel);

    uchar *d_img, *d_out_1, *d_out_2;
    float *d_kernel;
    cudaMalloc(&d_img, img.rows*img.cols*sizeof(uchar));
    cudaMalloc(&d_out_1, img.rows*img.cols*sizeof(uchar));
    cudaMalloc(&d_out_2, img.rows*img.cols*sizeof(uchar));
    cudaMalloc(&d_kernel, KER*KER*sizeof(float));

    cudaMemcpy(d_img, img.data, img.rows*img.cols*sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, kernel, KER*KER*sizeof(float));

    dim3 block(32, 32);
    dim3 grid((img.cols+block.x-1)/block.x, (img.rows+block.y-1)/block.y);

    uint64_t start = nanos();
    smart_device_convolution<<<grid, block>>>(d_img, d_out_1, img.cols, img.rows);
    cudaDeviceSynchronize();
    uint64_t end = nanos();
    std::cout << "Smart convolution time: " << (end-start) << " ns" << std::endl;

    cudaMemcpy(out_1.data, d_out_1, img.rows*img.cols*sizeof(uchar), cudaMemcpyDeviceToHost);

    cv::imshow("img", img);
    cv::imshow("out1", out_1);
    cv::waitKey(0);

}