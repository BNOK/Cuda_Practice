//this file is for testing different basic algorithms of CUDA 
// this is an implementation of everything learned through the udacity course 
// CS344
//Hope you enjoy 
#include <iostream>
#include <stdio.h>

using namespace std;

#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

// defined parametres
#define ARRAY_SIZE 64

//Functions 

void PreProcessing(int* h_in, int* h_out, int** d_in, int** d_out, int size);
void FillandDisplay(int* d_in, int* d_out, int size);

//Kernels

__global__ void DisplayArray(int* input, int size);
__global__ void FillArray(int* input, int size);
__global__ void SquareKernel(int* d_in, int* d_out, int size);

int main() {
	int inputArray[ARRAY_SIZE], outputArray[ARRAY_SIZE];
	int* d_in, *d_out;

	PreProcessing(inputArray, outputArray, &d_in, &d_out, ARRAY_SIZE);
	//checking code
	FillandDisplay(d_in, d_out, ARRAY_SIZE);
	



	

	return 0;
}


//CPU FUNCTIONS 
void PreProcessing(int* h_in, int* h_out, int** d_in, int** d_out, int size) {

	int byteSize = size * sizeof(int);
	//allocate memy for host vars
	memset(h_in, 0, byteSize);
	memset(h_out, 0, byteSize);

	//allocating memy for device vars 
	cudaMalloc(d_in, byteSize);
	cudaMalloc(d_out, byteSize);

	cudaMemcpy(*d_in, h_in, byteSize, cudaMemcpyHostToDevice);
	cudaMemset(*d_out, 0, byteSize);
}

void FillandDisplaySquare(int* d_in, int* d_out, int size) {

	dim3 GridSize(size, 1, 1);
	dim3 BlockSize(size, 1, 1);

	FillArray << <GridSize, BlockSize >> > (d_in,size);
	SquareKernel << <GridSize, BlockSize >> > (d_in, d_out, size);
	DisplayArray << <GridSize, BlockSize >> > (d_out,size);
}


//GPU KERNELS

//squaring numbers Kernel (MAP operations)

__global__ void SquareKernel(int* d_in, int* d_out, int size) {
	// considering we have a (n,1,1) grid and (n,1,1) block (x coordinates)
	int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < size) {
		int square = d_in[tId] * d_in[tId];
		d_out[tId] = square;
	}
}

__global__ void CubeKernel(int* d_in, int* d_out) {
	//considering we have a (1,1,1) grid and (n,1,1) block (x coordinates)
	int tId = threadIdx.x;
	int cube = d_in[tId] * d_in[tId] * d_in[tId];

	d_out[tId] = cube;
}

__global__ void FillArray(int* input, int size) {
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if (tId < size) {
		input[tId] = tId;
	}
	
}


// Displaying results
__global__ void DisplayArray(int* input, int size) {
	int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < size) {
		printf("arr[i] = %d | ", input[tId]);
	}
}