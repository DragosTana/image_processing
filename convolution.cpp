//
// Created by dragos on 10/01/23.
//
#include "msc.cpp"


Mat convolution(Mat& redChannel, Mat& greenChannel, Mat& blueChannel, int kernel[3][3]){
    Mat imgR(512,512, CV_8UC3);
    Mat imgG(512,512, CV_8UC3);
    Mat imgB(512,512, CV_8UC3);

    int row = 512;
    int cols = 512;

    for(int i = 1; i < row-1; i++){
        for(int j = 1; j < cols - 1; j++){

        }
    }

}