
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector_types.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

uchar4* converting_UCHAR_UCHAR4(uchar* input, int size);

void DisplayUchar4(uchar4* arr, int size);
void DisplayUchar(uchar* arr, int size);

__global__ void addKernel(uchar4* a, uchar* b)
{ 
        int threadId = blockIdx.x* blockDim.x* blockDim.y
                        + threadIdx.y * blockDim.x + threadIdx.x;

        b[threadId] = a[threadId].x + a[threadId].y + a[threadId].z;
}

int main()
{
    //reading image and getting data 
    string file_path = "C:/Users/Mega-Pc/Desktop/git-project/Cuda_Practice/Another_Desperate_Code/Test_Image.png";
    Mat img = imread(file_path, -1);
    uchar* imgData = img.data;

    // image dimensions
    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels();
    
    //sizes and numbers or bytes for uchar and uchar4
    int imageSize = rows * cols * channels;
    int BYTE_SIZE = imageSize * sizeof(uchar);
    
    cout << " after : " << endl;
    int imageSize_4 = rows * cols;
    int BYTE_SIZE_4 = imageSize_4 * sizeof(uchar4);
    
    //host output variable
    uchar* out_imgData = (uchar*)malloc(imageSize_4 * sizeof(uchar));

    //input data after converting to uchar4
    uchar4* out = (uchar4*)malloc(BYTE_SIZE_4);
    out = converting_UCHAR_UCHAR4(imgData, imageSize_4);
    DisplayUchar4(out, imageSize_4);



    //device variables
    uchar4* di_imgData;
    uchar* do_imgData;

    //allocating memory
    cudaMalloc(&di_imgData, BYTE_SIZE_4);
    cudaMalloc(&do_imgData, BYTE_SIZE / channels);
   
    //copying content
    cudaMemcpy(di_imgData, out, BYTE_SIZE_4, cudaMemcpyHostToDevice);

    //configuring kernel
    dim3 GridSize(1,1, 1);
    dim3 BlockSize(rows, cols, 1);
    addKernel << <GridSize, BlockSize >> > (di_imgData, do_imgData);

    //returning the results
    cudaMemcpy(out_imgData, do_imgData, BYTE_SIZE / channels, cudaMemcpyDeviceToHost);

    //printing results
    //namedWindow("color image", WINDOW_AUTOSIZE);
    for (int i = 0; i < imageSize_4; i++) {
        printf("out[%d] = %d \n", i, out_imgData[i]);
    }

    // Show the image inside it.
    //imshow("color image", img);



    // Wait for a keystroke.   
    waitKey(0);

    // Destroys all the windows created                         
    //destroyAllWindows();
    

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

uchar4* converting_UCHAR_UCHAR4(uchar* input ,int size_4) {
    uchar4* output = (uchar4*)malloc(size_4 * sizeof(uchar4));
    memset(output, 100, size_4 * sizeof(uchar4));
    
    for (int i = 0,j=0;j < size_4; i+=4,j++) {
        output[j].x = input[i];
        output[j].y = input[i+1];
        output[j].z = input[i+2];
        output[j].w = input[i+3];
        cout << "i = " << i << ", j = "<< j << endl;
    }

    return output;
}

uchar* converting_UCHAR4_UCHAR(uchar4* input, int size_1) {
    uchar* output = (uchar*)malloc(size_1 * sizeof(uchar));
    memset(output, 100, size_1 * sizeof(uchar));
    
    for (int i = 0, j = 0; i < size_1; i += 4, j++) {
        output[j] = input[i].x;
        output[j+1] = input[i].y;
        output[j+2] = input[i].z;
        output[j+3] = input[i].w;
        cout << "i = " << i << ", j = " << j << endl;
    }

    return output;
}

void DisplayUchar4(uchar4* arr,int size) {

    for (int i = 0; i < size; i++) {
        printf("arr[%d].x = %d , arr[%d].y = %d , arr[%d].z = %d \n", i, arr[i].x, i, arr[i].y, i, arr[i].z);   
    }
}

void DisplayUchar(uchar* arr, int size)
{
    // small display
    for (int i = 0; i < size; i++) {
        printf("imgData[%d] = %d | ", i, arr[i]);
    }
    cout << endl;
}
