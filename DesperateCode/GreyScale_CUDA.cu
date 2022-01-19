
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector_types.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>



__global__ void Conversion(uchar* d_in, uchar* d_out, int cols, int rows,  int channels) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	//Compute for only those threads which map directly to 
	//image grid
		int grey_offset = row * cols + col;
		int rgb_offset = grey_offset * channels;

		unsigned char r = d_in[rgb_offset + 0];
		unsigned char g = d_in[rgb_offset + 1];
		unsigned char b = d_in[rgb_offset + 2];

		d_out[grey_offset] = r * 0.299f + g * 0.587f + b * 0.114f;


}


void ConvertToGreyScale(uchar* hout_Data, uchar* hin_Data, int rows, int cols, int channels){


	//size in bytes
	int BYTE_SIZE = rows * cols * channels * sizeof(uchar4);

	//creating GPU pointers 
	uchar* d_in;
	uchar* d_out;

	//allocating memory
	cudaMalloc(&d_in, BYTE_SIZE);
	cudaMalloc(&d_out, BYTE_SIZE);
	cudaMemset(d_out, 0, BYTE_SIZE);

	//copying data into the GPU 
	cudaMemcpy(d_in, hin_Data, BYTE_SIZE, cudaMemcpyHostToDevice);

	// configuring and launching kernerl 
	dim3 GridSize(1,1,1);
	dim3 BlockSize(1,1,1);

	Conversion << <GridSize, BlockSize >> > (d_in, d_out, cols, rows, channels);

	//copying data to the host
	cudaMemcpy(hout_Data, d_out, BYTE_SIZE, cudaMemcpyDeviceToHost);

	//printing the results

}