#include <iostream>
#include <stdio.h>
#include <vector_types.h>
#include "H_D_Interface.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include 
using namespace std;
using namespace cv;


uchar4* UcharToUchar4( uchar* input_Data, int size, int channels);


int main(int* argc, char** argv) {
	// reading Image
	string File_Path = "C:/Users/Mega-Pc/Desktop/git-project/Cuda_Practice/DesperateCode/test_image.png";
	Mat inputImage;
	inputImage = imread(File_Path, -1);
	Mat rgbaImage;
	cvtColor(inputImage, rgbaImage, COLOR_BGRA2RGBA);

	//showing image
	imshow("image", inputImage);

	// collecting data
	uchar* imageData = inputImage.data;
	int r = inputImage.rows;
	int c = inputImage.cols;
	int channels = inputImage.channels();

	/*for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			int index = c * i + j;
			int channel = 0;
			cout << (int)imageData[index] << '|';
		}
		cout << endl;
	}*/
	

	//calling CUDA function
	uchar* greyImageData = (uchar*) malloc(sizeof(uchar) * r *c *channels);

	ConvertToGreyScale(greyImageData ,imageData, r, c, channels);


	waitKey(0);
	return 0;
}


//uchar4* UcharToUchar4(uchar* input_Data, int size, int channels)
//{
//	uchar4* out_image = (uchar4) malloc(sizeof(uchar4));
//
//	for (int i = 0; i < 16; channels +i ) {
//		out_image[i].x = input_Data[i];
//	}
//
//	return 
//}

//uchar3* UcharToUchar3(uchar* input_Data, int size, int channels)
//{
//	uchar3* out_image = nullptr;
//
//	for (int i = 0; i < size; i + channels) {
//		out_image[i].x = input_Data[i];
//		out_image[i].y = input_Data[i];
//		out_image[i].z = input_Data[i];
//	}
//
//	return out_image;
//}



