// OpenCV libs
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
// CUDA libs
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
// C/C++ libs
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 1024
#define GRID_SIZE 1024
#define ARRAY_SIZE 1024
using namespace std;


__global__ void ReduceKernel(int* d_a, int* d_b, int size);
__global__ void FillArray(int* input);
__global__ void DisplayArray(int* input);

int main() {
	int byteSize = ARRAY_SIZE * sizeof(int);
	int *a = (int*) malloc(byteSize );
	int *b = (int*) malloc(byteSize );

	memset(a, 0, byteSize);
	memset(b, 0, byteSize);
	
	int* d_a;
	int* d_b;
	
	cudaMalloc(&d_a, byteSize );
	cudaMalloc(&d_b, byteSize );
	cudaGetLastError();
	
	cudaMemcpy(d_a, a, byteSize , cudaMemcpyHostToDevice);
	cudaMemset(d_b, 0, byteSize );
	cudaGetLastError();
	
	dim3 GridSize(1024, 1, 1);
	dim3 blockSize(1024, 1, 1);

	FillArray << <GridSize, blockSize >> > (d_a);
	
	DisplayArray << <GridSize, blockSize >> > (d_a);

	ReduceKernel << <GridSize, blockSize >> > (d_a, d_b,byteSize);

	cudaMemcpy(b, d_b, byteSize , cudaMemcpyDeviceToHost);
	
	// host code 

	printf("b[0] = %d \n", b[0]);


	cv::waitKey(0);
	return 0;
}

__global__ void ReduceKernelShared(int* d_in, int* d_out) {
	// make a var in the shared mem
	extern __shared__ int s_data[];

	int g_id = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	s_data[tid] = d_in[g_id];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {

		if (tid < s) {
			d_in[g_id] += d_in[g_id + s];
		}
		//__syncthreads();
	}

	if (tid == 0) {
		d_out[blockIdx.x] = d_in[g_id];
	}
}


__global__ void ReduceKernel(int* d_a, int* d_b) {
	int gId = blockDim.x * blockIdx.x + threadIdx.x;
	int tId = threadIdx.x;

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {

		if (tId < s) {
			d_a[gId] += d_a[gId + s];
		}
		//__syncthreads();
	}
	
	if (tId == 0) {
		d_b[blockIdx.x] = d_a[gId];
	}
}

__global__ void FillArray(int* input) {
	int gId = blockDim.x * blockIdx.x + threadIdx.x;
	if (gId >= ARRAY_SIZE)
		return;
	input[gId] = gId;
}

__global__ void DisplayArray(int* input) {
	int gId = blockDim.x * blockIdx.x + threadIdx.x;
	if (gId >= ARRAY_SIZE)
		return;
	printf("input[%d] = %d | ", gId ,input[gId]);

	__syncthreads();
}
