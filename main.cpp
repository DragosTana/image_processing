//
// Created by dragos on 12/10/22.
//

#include<iostream>
#include<unistd.h>
#include "convolution.cpp"


int main(){

    std::string path = std::string(get_current_dir_name()) + std::string("/Images/Lena.png");
//    std::cout<<path<<std::endl;

    Mat img = imread(path, IMREAD_COLOR);

    Mat imgR(512,512, CV_8UC3);
    Mat imgG(512,512, CV_8UC3);
    Mat imgB(512,512, CV_8UC3);

    MatType(img);

    int kernel[3][3] = { 1/16, 2/16, 1/16,
                         2/16, 4/16, 2/16,
                         1/16, 2/16, 1/16 };

    divide_channels(img, imgR, imgG, imgB);

    imshow("Red", imgR);
    imshow("Green", imgG);
    imshow("Blue", imgB);
    imshow("Lena", img);
    waitKey(0);
    return 0;

}