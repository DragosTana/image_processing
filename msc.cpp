//
// Created by dragos on 10/01/23.
//

#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

using namespace cv;

void MatType( Mat& inputMat ){

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

void divide_channels(const Mat& img, Mat& imgR, Mat& imgG, Mat& imgB){

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            //std::cout << img.at<Vec3b>(i,j)[0] << " " << img.at<Vec3b>(i,j)[1] << " " << img.at<Vec3b>(i,j)[2] << std::endl;
            imgB.at<Vec3b>(i,j)[0] = img.at<Vec3b>(i,j)[0];
            imgG.at<Vec3b>(i,j)[1] = img.at<Vec3b>(i,j)[1];
            imgR.at<Vec3b>(i,j)[2] = img.at<Vec3b>(i,j)[2];
        }
    }
}

Mat merge_channels (const Mat& imgR, const Mat& imgG, const Mat& imgB){

    Mat img(512, 512, CV_8SC3);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            //std::cout << img.at<Vec3b>(i,j)[0] << " " << img.at<Vec3b>(i,j)[1] << " " << img.at<Vec3b>(i,j)[2] << std::endl;
            img.at<Vec3b>(i,j)[0] = imgB.at<Vec3b>(i,j)[0];
            img.at<Vec3b>(i,j)[1] = imgG.at<Vec3b>(i,j)[1];
            img.at<Vec3b>(i,j)[2] = imgR.at<Vec3b>(i,j)[2];
        }
    }
    return(img);
}

Mat