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

    double kernel[3][3] = {{1, 2, 1},
                           {2, 4, 2},
                           {1, 2, 1}};

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j ++){
            kernel[i][j] = kernel[i][j]/16;
        }
    }

    divide_channels(img, imgR, imgG, imgB);

//    imshow("Green", imgG);
//    imshow("Blue", imgB);
//    imshow("Lena", img);

    Mat imgR_blur(512, 512, CV_8SC3);
    imgR_blur = convolution(imgR, imgG, imgB, kernel);
    imshow("Red", imgR);
    imshow("Red Blur",imgR_blur);
    waitKey(0);

    return 0;

}