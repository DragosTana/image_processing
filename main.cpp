//
// Created by dragos on 12/10/22.
//

#include<iostream>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

using namespace cv;

void MatType( Mat& inputMat )
{
    int inttype = inputMat.type();

    std::string r, a;
    uchar depth = inttype & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (inttype >> CV_CN_SHIFT);
    switch ( depth ) {
        case CV_8U:  r = "8U";   a = "Mat.at<uchar>(y,x)"; break;
        case CV_8S:  r = "8S";   a = "Mat.at<schar>(y,x)"; break;
        case CV_16U: r = "16U";  a = "Mat.at<ushort>(y,x)"; break;
        case CV_16S: r = "16S";  a = "Mat.at<short>(y,x)"; break;
        case CV_32S: r = "32S";  a = "Mat.at<int>(y,x)"; break;
        case CV_32F: r = "32F";  a = "Mat.at<float>(y,x)"; break;
        case CV_64F: r = "64F";  a = "Mat.at<double>(y,x)"; break;
        default:     r = "User"; a = "Mat.at<UKNOWN>(y,x)"; break;
    }
    r += "C";
    r += (chans+'0');
    std::cout << "Mat is of type " << r << " and should be accessed with " << a << std::endl;

}


Mat GaussianBlur(Mat& redChannel, Mat& greenChannel, Mat& blueChannel, int kernel[3][3]){
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


int main(){
    std::string path = "/home/dragos/Projects/Convolution/Lena.png";

    Mat img = imread(path, IMREAD_COLOR);

    Mat imgR(512,512, CV_8UC3);
    Mat imgG(512,512, CV_8UC3);
    Mat imgB(512,512, CV_8UC3);

    MatType(img);

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
        //std::cout << img.at<Vec3b>(i,j)[0] << " " << img.at<Vec3b>(i,j)[1] << " " << img.at<Vec3b>(i,j)[2] << std::endl;
        imgB.at<Vec3b>(i,j)[0] = img.at<Vec3b>(i,j)[0];
        imgG.at<Vec3b>(i,j)[1] = img.at<Vec3b>(i,j)[1];
        imgR.at<Vec3b>(i,j)[2] = img.at<Vec3b>(i,j)[2];
        }
    }

    int kernel[3][3] = { 1/16, 2/16, 1/16,
                         2/16, 4/16, 2/16,
                         1/16, 2/16, 1/16 };

    imshow("Red", imgR);
    imshow("Green", imgG);
    imshow("Blue", imgB);
    imshow("Lena", img);
    waitKey(0);
    return 0;

}