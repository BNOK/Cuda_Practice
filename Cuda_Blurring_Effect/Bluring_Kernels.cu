//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
//
//#include <iostream>
//
//using namespace std;
//
//void Clamp(int& pos, int maxPos) {
//    pos = pos < 0 ? 0 : pos;
//    pos = pos > maxPos ? maxPos : pos;
//}
//
//__global__
//void gaussian_blur(const unsigned char* const inputChannel,
//    unsigned char* const outputChannel,
//    int numRows, int numCols,
//    const float* const filter, const int filterWidth)
//{
//    // TODO
//
//    // NOTE: Be sure to compute any intermediate results in floating point
//    // before storing the final result as unsigned char.
//
//    // NOTE: Be careful not to try to access memory that is outside the bounds of
//    // the image. You'll want code that performs the following check before accessing
//    // GPU memory:
//    //
//    // if ( absolute_image_position_x >= numCols ||
//    //      absolute_image_position_y >= numRows )
//    // {
//    //     return;
//    // }
//
//    // NOTE: If a thread's absolute position 2D position is within the image, but some of
//    // its neighbors are outside the image, then you will need to be extra careful. Instead
//    // of trying to read such a neighbor value from GPU memory (which won't work because
//    // the value is out of bounds), you should explicitly clamp the neighbor values you read
//    // to be within the bounds of the image. If this is not clear to you, then please refer
//    // to sequential reference solution for the exact clamping semantics you should follow.
//
//    //finding 2Dpos 
//    int2 threadPos_2D = make_int2(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
//
//    if (threadPos_2D.x > numRows || threadPos_2D.y > numCols)
//        return;
//
//    int threadPos_1D = threadPos_2D.y * numCols + threadPos_2D.x;
//
//    const int halfWidth = filterWidth / 2;
//
//    float finalVal = 0.0f;
//    for (int x = 0; x < filterWidth; x++) {
//        for (int y = 0; y < filterWidth; y++) {
//            
//            int image_r = static_cast<int>(threadPos_2D.y + (y - halfWidth));
//            Clamp(image_r, numRows);
//
//            int image_c = static_cast<int>(threadPos_2D.x + (x - halfWidth));
//            Clamp(image_c, numCols);
//
//
//            finalVal += filter[y * filterWidth + x] * inputChannel[image_r * numCols + image_c];
//        }
//    }
//
//
//    
//    outputChannel[threadPos_1D] += finalVal;
//}
//
//
//__global__
//void separateChannels(const uchar4* const inputImageRGBA,
//    int numRows,
//    int numCols,
//    unsigned char* const redChannel,
//    unsigned char* const greenChannel,
//    unsigned char* const blueChannel)
//{
//    // TODO
//    //
//    // NOTE: Be careful not to try to access memory that is outside the bounds of
//    // the image. You'll want code that performs the following check before accessing
//    // GPU memory:
//    //
//    // if ( absolute_image_position_x >= numCols ||
//    //      absolute_image_position_y >= numRows )
//    // {
//    //     return;
//    // }
//
//    int2 threadPos_2D = make_int2(blockIdx.x * numCols + threadIdx.x, blockIdx.y * numRows + threadIdx.y);
//
//    if (threadPos_2D.x > numCols || threadPos_2D.y > numRows)
//        return;
//
//    int thread_1d_pos = threadPos_2D.y * numCols + threadPos_2D.x;
//    redChannel[thread_1d_pos] = inputImageRGBA[thread_1d_pos].x;
//    greenChannel[thread_1d_pos] = inputImageRGBA[thread_1d_pos].y;
//    blueChannel[thread_1d_pos] = inputImageRGBA[thread_1d_pos].z;
//}
//
