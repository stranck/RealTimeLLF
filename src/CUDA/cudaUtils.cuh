#pragma once

#include <stdint.h>
#include "../utils/structs.h"
#include "cudaStructs.cuh"
#ifdef CUDA_INCLUDE
	#include <cuda.h>
	#include <cuda_runtime.h>
#else
    //Shitty hack to remove the errors from vscode even if it doesn't detect my CUDA installation
    #define __global__
    #define __shared__
	#define __device__
    #define cudaError_t int
    #define cudaMemcpyHostToDevice 1
    #define cudaSuccess 0
    #define cudaMalloc(a, b)({int disajiodjsao = 0; disajiodjsao;})
    #define cudaMemcpy(a, b, c, d)({int dkosapkogkg = 0; dkosapkogkg;})
    #define cudaGetErrorString(e)({"asd";})
    #define __syncthreads(){}
    #define blockDim zero3i32
    #define threadIdx zero3i32
#endif


#define CHECK(call){                    \
    const cudaError_t error = call;     \
    if(error != cudaSuccess){           \
        printf("Error: %s:%d", __FILE__, __LINE__); \
        printf("code: %d, reason %s\n", error, cudaGetErrorString(error)); \
        exit(1);                        \
    }                                   \
}

#define KERNEL_DIMENSION 5
Kernel createFilterDevice();
Image3 * makeImage3Device(uint32_t width, uint32_t height);
Image3 * copyImg3Host2Device(Image3 * h_img);
Pyramid createPyramidDevice(uint32_t width, uint32_t height, uint8_t nLevels);
__device__ void d_imgcpy3(Image3 *d_dest, Image3 *d_source);
__device__ void d_subimage3(Image3 *dest, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY);
__device__ void d_remap(Image3 * img, const Pixel3 g0, double sigma, double alpha, double beta);