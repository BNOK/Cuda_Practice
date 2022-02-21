//
//// OpenCV libs
//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
//// CUDA libs
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//// C/C++ libs
//#include <stdio.h>
//#include <iostream>
//#include "utils.h"
//
//#define TILE 16
//
//
////namespaces
//using namespace std;
//using namespace cv;
//
//// FUNCTIONS DEFINITION 
//void PreProcess(string path_file, uchar4** h_inputimage, uchar4** h_outputimage, uchar4** d_inputimage, uchar4** d_outputimage, uchar* d_red, uchar* d_green, uchar* d_blue, int* rows,int* cols);
//void AllocatingMemy(uchar* arr1, uchar* arr2, uchar* arr3, int size);
//// KERNEL DEFINITION
//__global__ void display(uchar* arr);
//__global__ void Bluring_Kernel();
//__global__ void Seperate_Colors(const uchar4* const inputImageRGBA,
//    int numRows,
//    int numCols,
//    unsigned char* const redChannel,
//    unsigned char* const greenChannel,
//    unsigned char* const blueChannel);
//
//cv::Mat imageInputRGBA;
//cv::Mat imageOutputRGBA;
//
//uchar4* d_inputImageRGBA__;
//uchar4* d_outputImageRGBA__;
//
//float* h_filter__;
//
//size_t numRows() { return imageInputRGBA.rows; }
//size_t numCols() { return imageInputRGBA.cols; }
//
//
//int main()
//{
//    // Necessary vars
//    string path_file = "C:/Users/Mega-Pc/Desktop/git-project/Cuda_Practice/Cuda_Blurring_Effect/Test_Image.png";
//    uchar4* h_inputImageRGBA, * d_inputImageRGBA;
//    uchar4* h_outputImageRGBA, * d_outputImageRGBA;
//    int rows, cols;
//    uchar* d_red = 0, * d_green = 0, * d_blue = 0;
//    
//    PreProcess(path_file, &h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA, d_red, d_green, d_blue, &rows, &cols);
//    printf("hello 1");
//
//    
//
//    uchar* d_redBlurred, * d_greenBlurred, * d_blueBlurred;
//
//    
//   
//
//    //printf("rows = %d , cols = %d", rows, cols);
//    //// kernel configuration
//    //dim3 BlockDim(TILE, TILE, 1);
//    //dim3 GridDim(cols/TILE, rows/TILE, 1);
//
//    //Seperate_Colors << <GridDim, BlockDim >> > (h_inputImageRGBA, rows, cols, d_red, d_green, d_blue);
//
//    waitKey(0);
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
////cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
////{
////    int *dev_a = 0;
////    int *dev_b = 0;
////    int *dev_c = 0;
////    cudaError_t cudaStatus;
////
////    // Choose which GPU to run on, change this on a multi-GPU system.
////    cudaStatus = cudaSetDevice(0);
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
////        goto Error;
////    }
////
////    // Allocate GPU buffers for three vectors (two input, one output)    .
////    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaMalloc failed!");
////        goto Error;
////    }
////
////    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaMalloc failed!");
////        goto Error;
////    }
////
////    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaMalloc failed!");
////        goto Error;
////    }
////
////    // Copy input vectors from host memory to GPU buffers.
////    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaMemcpy failed!");
////        goto Error;
////    }
////
////    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaMemcpy failed!");
////        goto Error;
////    }
////
////    // Launch a kernel on the GPU with one thread for each element.
////    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
////
////    // Check for any errors launching the kernel
////    cudaStatus = cudaGetLastError();
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
////        goto Error;
////    }
////    
////    // cudaDeviceSynchronize waits for the kernel to finish, and returns
////    // any errors encountered during the launch.
////    cudaStatus = cudaDeviceSynchronize();
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
////        goto Error;
////    }
////
////    // Copy output vector from GPU buffer to host memory.
////    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaMemcpy failed!");
////        goto Error;
////    }
////
////Error:
////    cudaFree(dev_c);
////    cudaFree(dev_a);
////    cudaFree(dev_b);
////    
////    return cudaStatus;
////}
//
//// utils functions 
//
//void preProcess(uchar4** h_inputImageRGBA, uchar4** h_outputImageRGBA,
//    uchar4** d_inputImageRGBA, uchar4** d_outputImageRGBA,
//    unsigned char** d_redBlurred,
//    unsigned char** d_greenBlurred,
//    unsigned char** d_blueBlurred,
//    float** h_filter, int* filterWidth,
//    const std::string& filename) {
//
//    //make sure the context initializes ok
//    checkCudaErrors(cudaFree(0));
//
//    cv::Mat image = cv::imread(filename.c_str(), IMREAD_COLOR);
//    if (image.empty()) {
//        std::cerr << "Couldn't open file: " << filename << std::endl;
//        exit(1);
//    }
//
//    cv::cvtColor(image, imageInputRGBA, COLOR_BGR2RGBA);
//
//    //allocate memory for the output
//    imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);
//
//    //This shouldn't ever happen given the way the images are created
//    //at least based upon my limited understanding of OpenCV, but better to check
//    if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
//        std::cerr << "Images aren't continuous!! Exiting." << std::endl;
//        exit(1);
//    }
//
//    *h_inputImageRGBA = (uchar4*)imageInputRGBA.ptr<unsigned char>(0);
//    *h_outputImageRGBA = (uchar4*)imageOutputRGBA.ptr<unsigned char>(0);
//
//    const size_t numPixels = numRows() * numCols();
//    //allocate memory on the device for both input and output
//    checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
//    checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
//    checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4))); //make sure no memory is left laying around
//
//    //copy input array to the GPU
//    checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
//
//    d_inputImageRGBA__ = *d_inputImageRGBA;
//    d_outputImageRGBA__ = *d_outputImageRGBA;
//
//    //now create the filter that they will use
//    const int blurKernelWidth = 9;
//    const float blurKernelSigma = 2.;
//
//    *filterWidth = blurKernelWidth;
//
//    //create and fill the filter we will convolve with
//    *h_filter = new float[blurKernelWidth * blurKernelWidth];
//    h_filter__ = *h_filter;
//
//    float filterSum = 0.f; //for normalization
//
//    for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) {
//        for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) {
//            float filterValue = expf(-(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
//            (*h_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c + blurKernelWidth / 2] = filterValue;
//            filterSum += filterValue;
//        }
//    }
//
//    float normalizationFactor = 1.f / filterSum;
//
//    for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) {
//        for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) {
//            (*h_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c + blurKernelWidth / 2] *= normalizationFactor;
//        }
//    }
//
//    //blurred
//    checkCudaErrors(cudaMalloc(d_redBlurred, sizeof(unsigned char) * numPixels));
//    checkCudaErrors(cudaMalloc(d_greenBlurred, sizeof(unsigned char) * numPixels));
//    checkCudaErrors(cudaMalloc(d_blueBlurred, sizeof(unsigned char) * numPixels));
//    checkCudaErrors(cudaMemset(*d_redBlurred, 0, sizeof(unsigned char) * numPixels));
//    checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
//    checkCudaErrors(cudaMemset(*d_blueBlurred, 0, sizeof(unsigned char) * numPixels));
//}
//
//void AllocatingMemy(uchar* arr1, uchar* arr2, uchar* arr3,int size) {
//    int BYTE_SIZE = size * sizeof(uchar);
//
//    checkCudaErrors(cudaMalloc(&arr1, BYTE_SIZE));
//    checkCudaErrors(cudaMalloc(&arr2, BYTE_SIZE));
//    checkCudaErrors(cudaMalloc(&arr3, BYTE_SIZE));
//
//    checkCudaErrors(cudaMemset(arr1, 1, BYTE_SIZE));
//    checkCudaErrors(cudaMemset(arr2, 2, BYTE_SIZE));
//    checkCudaErrors(cudaMemset(arr3, 3, BYTE_SIZE));
//}
//// KERNEL IMPLEMENTATION
//__global__ void display(uchar* arr) {
//    cout << "hello world !" << endl;
//    int threadId = threadIdx.x;
//    int element = arr[threadId];
//    cout<<"arr[i] = % d" << element<<endl;
//}
//
//__global__ void Bluring_Kernel()
//{
//
//}
//
//__global__ void Seperate_Colors(const uchar4* const inputImageRGBA,
//    int numRows,
//    int numCols,
//    unsigned char* const redChannel,
//    unsigned char* const greenChannel,
//    unsigned char* const blueChannel) {
//
//    //thread index x and y 
//    int Xid = blockIdx.x * blockDim.x + threadIdx.x;
//    int Yid = blockIdx.y * blockDim.y + threadIdx.y;
//    // thread index 
//    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
//    int threadId = blockId * (blockDim.x * blockDim.y)
//        + (threadIdx.y * blockDim.x) + threadIdx.x;
//
//    if (Xid > numRows || Yid > numCols)
//        return;
//
//    redChannel[threadId] = inputImageRGBA[threadId].x;
//    greenChannel[threadId] = inputImageRGBA[threadId].y;
//    blueChannel[threadId] = inputImageRGBA[threadId].z;
//
//    
//}
//
//void DisplayArray(unsigned char* arr, int size) {
//    for (int i = 0; i < size; i++) {
//        printf("arr[i] = %d | ", arr[i]);
//    }
//}
