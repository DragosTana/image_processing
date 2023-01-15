//
// Created by dragos on 12/10/22.
//

#include<iostream>
#include<unistd.h>
#include "convolution.cpp"

using namespace std;

int main(){
    Mat img;
    Mat img_conv;
    Mat my_kernel;
    Mat my_conv;

    // Controlling if the image is loaded correctly
    std::string path = std::string(get_current_dir_name()) + std::string("/Images/Lena.png");
    img = imread(path, IMREAD_COLOR);
    if(! img.data )
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    imshow("original image", img);
    img.convertTo(img, CV_64FC3);

    int kernel_size;   // permitted sizes: 3, 5, 7, 9 etc
    cout << "Select the size of kernel (odd number from 3): \n" << endl;
    cin >> kernel_size;

    // Defining the kernel here
    int selection;
    cout << "Select the type of kernel:\n" << "1. Identity Operator \n2. Mean Filter \n3. Spatial shift \n4. Sharpening\n-> ";
    cin >> selection;
    switch (selection){
        case 1:
            my_kernel = (Mat_<double>(kernel_size,kernel_size) << 0, 0, 0, 0, 1, 0, 0, 0, 0);
            break;
        case 2:
            my_kernel = (Mat_<double>(kernel_size,kernel_size) << 1, 1, 1, 1, 1, 1, 1, 1, 1) / ( kernel_size * kernel_size);
            break;
        case 3:
            my_kernel = (Mat_<double>(kernel_size,kernel_size) << 0, 0, 0, 0, 0, 1, 0, 0, 0);
            break;
        case 4:
            my_kernel = (Mat_<double>(kernel_size,kernel_size) << -1, -1, -1, -1, 17, -1, -1, -1, -1) / ( kernel_size * kernel_size);
            break;
        default:
            cerr << "Invalid selection";
            return 1;
            break;
    }
    cout << "my kernel:\n "<<my_kernel << endl;

    // Adding the countour of nulls around the original image, to avoid border problems during convolution
    img_conv = Mat(img.rows + my_kernel.rows - 1, img.cols + my_kernel.cols - 1, CV_64FC3, CV_RGB(0,0,0));

    for (int x=0; x<img.rows; x++) {
        for (int y=0; y<img.cols; y++) {
            img_conv.at<Vec3d>(x+1,y+1)[0] = img.at<Vec3d>(x,y)[0];
            img_conv.at<Vec3d>(x+1,y+1)[1] = img.at<Vec3d>(x,y)[1];
            img_conv.at<Vec3d>(x+1,y+1)[2] = img.at<Vec3d>(x,y)[2];
        }
    }

    //Performing the convolution
    my_conv = Mat(img.rows, img.cols, CV_64FC3, CV_RGB(0,0,0));
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
    my_conv.convertTo(my_conv, CV_8UC3);
    imshow("convolution - manual", my_conv);

    // Performing the filtering using the opencv funtions
    Mat dst;
    filter2D(img, dst, -1 , my_kernel, Point( -1, -1 ), 0, BORDER_DEFAULT );
    dst.convertTo(dst, CV_8UC3);
    imshow("convolution - opencv", dst);


    waitKey();
    return 0;

//
//    std::string path = std::string(get_current_dir_name()) + std::string("/Images/Lena.png");
////    std::cout<<path<<std::endl;
//
//    Mat img = imread(path, IMREAD_COLOR);
//
//    Mat imgR(512,512, CV_8UC3);
//    Mat imgG(512,512, CV_8UC3);
//    Mat imgB(512,512, CV_8UC3);
//
//    MatType(img);
//
//    double kernel[3][3] = {{1, 2, 1},
//                           {2, 4, 2},
//                           {1, 2, 1}};
//
//    for(int i = 0; i < 3; i++){
//        for(int j = 0; j < 3; j ++){
//            kernel[i][j] = kernel[i][j]/16;
//        }
//    }
//
//    divide_channels(img, imgR, imgG, imgB);
//
////    imshow("Green", imgG);
////    imshow("Blue", imgB);
////    imshow("Lena", img);
//
//    Mat img_blur(512, 512, CV_8SC3);
//    img_blur = convolution(imgR, imgG, imgB, kernel);
//    std::cin.ignore();
//
//    imshow("Lena", img);
//    imshow("Lena Blur",img_blur);
//    waitKey(0);
//
//    return 0;

}
