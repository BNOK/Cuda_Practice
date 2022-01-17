//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//
//
//
//__global__ void Karnel(int* d_a,int* d_b,int* d_c) {
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//
//	d_c[index] = d_a[index] + d_b[index];
//}
//
//
//
//
//
//int main(int argc, char** argv) {
//	//constant memory size
//	const int ARRAY_SIZE = 5;
//	const int BYTE_SIZE = ARRAY_SIZE * sizeof(int);
//	//declaring host variables 
//	int a[ARRAY_SIZE] = { 1,2,3,4,5 };
//	int b[ARRAY_SIZE] = { 10,20,30,40,50 };
//	int c[ARRAY_SIZE];
//
//	//declaring GPU pointers
//	int* d_a;
//	int* d_b;
//	int* d_c;
//
//	//allocating memory
//	cudaMalloc(&d_a, BYTE_SIZE);
//	cudaMalloc(&d_b, BYTE_SIZE);
//	cudaMalloc(&d_c, BYTE_SIZE);
//
//	//copying data from Host to device
//	cudaMemcpy(d_a, a, BYTE_SIZE, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, b, BYTE_SIZE, cudaMemcpyHostToDevice);
//
//	//Launching Kernel
//	dim3 Grid_Size(1, 1, 1);
//	dim3 Block_Size(5, 1, 1);
//	Karnel << <Grid_Size, Block_Size >> > (d_a, d_b, d_c);
//
//	//copying the results to the CPU
//	cudaMemcpy(c, d_c, BYTE_SIZE, cudaMemcpyDeviceToHost);
//
//	for (int i = 0; i < ARRAY_SIZE; i++) {
//		printf("c[%d] = %d\n", i, c[i]);
//	}
//
//
//	return 0;
//}