
// OpenCV libs
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
// CUDA libs
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// C/C++ libs
#include <stdio.h>
#include <iostream>
#include "utils.h"

#define TILE 16


//namespaces
using namespace std;
using namespace cv;

// FUNCTIONS DEFINITION 
void PreProcess(string path_file, uchar4** h_inputimage, uchar4** h_outputimage, uchar4** d_inputimage, uchar4** d_outputimage, uchar* d_red, uchar* d_green, uchar* d_blue, int* rows,int* cols);
void AllocatingMemy(uchar* arr1, uchar* arr2, uchar* arr3, int size);
// KERNEL DEFINITION
__global__ void display(uchar* arr);
__global__ void Bluring_Kernel();
__global__ void Seperate_Colors(const uchar4* const inputImageRGBA,
    int numRows,
    int numCols,
    unsigned char* const redChannel,
    unsigned char* const greenChannel,
    unsigned char* const blueChannel);


int main()
{
    string path_file = "C:/Users/Mega-Pc/Desktop/git-project/Cuda_Practice/Cuda_Blurring_Effect/Test_Image.png";
    uchar4* h_inputImageRGBA, * d_inputImageRGBA;
    uchar4* h_outputImageRGBA, * d_outputImageRGBA;
    int rows, cols;
    uchar* d_red = 0, * d_green = 0, * d_blue = 0;
    
    PreProcess(path_file, &h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA, d_red, d_green, d_blue, &rows, &cols);
    printf("hello 1");
    display << <1, 1 >> > (d_red);

    uchar* d_redBlurred, * d_greenBlurred, * d_blueBlurred;

    
   

    //printf("rows = %d , cols = %d", rows, cols);
    //// kernel configuration
    //dim3 BlockDim(TILE, TILE, 1);
    //dim3 GridDim(cols/TILE, rows/TILE, 1);

    //Seperate_Colors << <GridDim, BlockDim >> > (h_inputImageRGBA, rows, cols, d_red, d_green, d_blue);

    waitKey(0);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}

// utils functions 
void PreProcess(string path_file, uchar4** h_inputimage,uchar4** h_outputimage, uchar4** d_inputimage, uchar4** d_outputimage, uchar* d_red, uchar* d_green, uchar* d_blue,int* rows, int* cols) {
    
    // reading Image
    Mat img = imread(path_file, IMREAD_COLOR);
    Mat rgbaImg;
    cvtColor(img, rgbaImg, COLOR_BGR2RGBA);
    *rows = rgbaImg.rows;
    *cols = rgbaImg.cols;
    int numPixels = rgbaImg.rows * rgbaImg.cols;

    *h_inputimage = (uchar4*)rgbaImg.ptr<unsigned char>(0);
    *h_outputimage = (uchar4*)rgbaImg.ptr<unsigned char>(0);
    

    checkCudaErrors(cudaMalloc(d_inputimage, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc(d_outputimage, sizeof(uchar4) * numPixels));
    checkCudaErrors( cudaMemset(*d_outputimage, 0, numPixels * sizeof(uchar4)));

    AllocatingMemy(d_red, d_green, d_blue, numPixels);
}

void AllocatingMemy(uchar* arr1, uchar* arr2, uchar* arr3,int size) {
    int BYTE_SIZE = size * sizeof(uchar);

    checkCudaErrors(cudaMalloc(&arr1, BYTE_SIZE));
    checkCudaErrors(cudaMalloc(&arr2, BYTE_SIZE));
    checkCudaErrors(cudaMalloc(&arr3, BYTE_SIZE));

    checkCudaErrors(cudaMemset(arr1, 1, BYTE_SIZE));
    checkCudaErrors(cudaMemset(arr2, 2, BYTE_SIZE));
    checkCudaErrors(cudaMemset(arr3, 3, BYTE_SIZE));
}
// KERNEL IMPLEMENTATION
__global__ void display(uchar* arr) {
    cout << "hello world !" << endl;
    int threadId = threadIdx.x;
    int element = arr[threadId];
    cout<<"arr[i] = % d" << element<<endl;
}

__global__ void Bluring_Kernel()
{

}

__global__ void Seperate_Colors(const uchar4* const inputImageRGBA,
    int numRows,
    int numCols,
    unsigned char* const redChannel,
    unsigned char* const greenChannel,
    unsigned char* const blueChannel) {

    //thread index x and y 
    int Xid = blockIdx.x * blockDim.x + threadIdx.x;
    int Yid = blockIdx.y * blockDim.y + threadIdx.y;
    // thread index 
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (Xid > numRows || Yid > numCols)
        return;

    redChannel[threadId] = inputImageRGBA[threadId].x;
    greenChannel[threadId] = inputImageRGBA[threadId].y;
    blueChannel[threadId] = inputImageRGBA[threadId].z;

    
}

void DisplayArray(unsigned char* arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("arr[i] = %d | ", arr[i]);
    }
}
