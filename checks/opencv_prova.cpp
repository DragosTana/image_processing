#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

// to compile run g++ opencv_prova.cpp -I /usr/local/include/opencv4/ -L /usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs

int main() {

    int size[] = {10, 10};

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
    
    return 0;

}
