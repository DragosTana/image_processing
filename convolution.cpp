//
// Created by dragos on 10/01/23.
//
#include "msc.cpp"

int matrix_multiplication(const double kernel[3][3], Mat& channel, int& x, int& y){
    //Input are the kernel, the channel and the coordinates of the point on which we perform the convolution.
    int value = 0;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j ++){
            value += (kernel[i][j])*int(channel.at<uchar>(x-1+j, y-1+i));
//            std::cout<<kernel[i][j]<<" "<< int(channel.at<uchar>(x-1+j, y-1+i))<<std::endl;
//            std::cout<<value<<std::endl;
        }
    }
//    std::cout<<"next pixel"<<std::endl;
    return(value);
}

void print_kernel(double kernel[3][3]){
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j ++){
            std::cout<<kernel[i][j]<<" ";
        }
    }
}

Mat convolution(Mat& redChannel, Mat& greenChannel, Mat& blueChannel, const double kernel[3][3]){
    Mat imgR(512,512, CV_8UC3);
    Mat imgG(512,512, CV_8UC3);
    Mat imgB(512,512, CV_8UC3);

    int row = 512;
    int cols = 512;
    int count = 0;
    for(int i = 1; i < row-1; i++){
        for(int j = 1; j < cols - 1; j++){
            imgR.at<uchar>(i, j) = char(matrix_multiplication(kernel, redChannel, i, j));
            //std::cout<< matrix_multiplication(kernel, redChannel, i, j);
        }
    }
    std::cout<<count;
    return imgR;
}