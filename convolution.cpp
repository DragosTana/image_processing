//
// Created by dragos on 10/01/23.
//
#include "msc.cpp"

void print_kernel(double kernel[3][3]){
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j ++){
            std::cout<<kernel[i][j]<<" ";
        }
    }
}

Mat convolutionV1(Mat& img, Mat my_kernel, Mat img_conv){

    //Performing the convolution
    Mat my_conv = Mat(img.rows, img.cols, CV_64FC3, CV_RGB(0,0,0));
    for (int x=(my_kernel.rows-1)/2; x<img_conv.rows-((my_kernel.rows-1)/2); x++) {
        for (int y=(my_kernel.cols-1)/2; y<img_conv.cols-((my_kernel.cols-1)/2); y++) {
            double comp_1=0;
            double comp_2=0;
            double comp_3=0;
            for (int u=-(my_kernel.rows-1)/2; u<=(my_kernel.rows-1)/2; u++) {
                for (int v=-(my_kernel.cols-1)/2; v<=(my_kernel.cols-1)/2; v++) {
                    comp_1 = comp_1 + ( img_conv.at<Vec3d>(x+u,y+v)[0] * my_kernel.at<double>(u + ((my_kernel.rows-1)/2) ,v + ((my_kernel.cols-1)/2)));
                    comp_2 = comp_2 + ( img_conv.at<Vec3d>(x+u,y+v)[1] * my_kernel.at<double>(u + ((my_kernel.rows-1)/2),v + ((my_kernel.cols-1)/2)));
                    comp_3 = comp_3 + ( img_conv.at<Vec3d>(x+u,y+v)[2] * my_kernel.at<double>(u +  ((my_kernel.rows-1)/2),v + ((my_kernel.cols-1)/2)));
                }
            }
            my_conv.at<Vec3d>(x-((my_kernel.rows-1)/2),y-(my_kernel.cols-1)/2)[0] = comp_1;
            my_conv.at<Vec3d>(x-((my_kernel.rows-1)/2),y-(my_kernel.cols-1)/2)[1] = comp_2;
            my_conv.at<Vec3d>(x-((my_kernel.rows-1)/2),y-(my_kernel.cols-1)/2)[2] = comp_3;
        }
    }
}


