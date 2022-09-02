#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdint.h>
#include "cuda.cuh"
#include "../utils/utils.h"
#include "../utils/structs.h"
#include "../utils/llfUtils.h"
#include "../utils/extramath.h"
#include "../utils/imageutils.h"
#ifdef CUDA_INCLUDE
	#include <cuda.h>
	#include <cuda_runtime.h>
#else
    //Shitty hack to remove the errors from vscode even if it doesn't detect my CUDA installation
    #define __global__
    #define __shared__
	#define __device__
	#define __host__
    #define cudaError_t int
	#define cudaMemcpyDeviceToHost 0
    #define cudaMemcpyHostToDevice 1
    #define cudaSuccess 0
    #define cudaMalloc(a, b)({int disajiodjsao = 0; disajiodjsao;})
    #define cudaMemcpy(a, b, c, d)({int dkosapkogkg = 0; dkosapkogkg;})
    #define cudaGetErrorString(e)({"asd";})
    #define __syncthreads(){}
	#define cudaDeviceSynchronize(){}
    #define blockDim zero3i32
    #define blockIdx zero3i32
    #define threadIdx zero3i32
	#define dim3 Vec3u32
	#define grid(x, y)({dim3{x, y, 0}});
#endif

#define CHECK(call){                    \
    const cudaError_t error = call;     \
    if(error != cudaSuccess){           \
        printf("Error: %s:%d", __FILE__, __LINE__); \
        printf("code: %d, reason %s\n", error, cudaGetErrorString(error)); \
        exit(1);                        \
    }                                   \
}

#define d_getPixel3(_pxs, _width, _x, _y){ \
	_pxs[(_y) * (_width) + (_x)] \
}
#define d_setPixel3(_pxs, _width, _x, _y, _px){ \
	_pxs[(_y) * (_width) + (_x)] = _px; \
}

__global__ void d_subimage3Test(Image3 *dest, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY);

#define KERNEL_DIMENSION 5

__host__ Kernel createFilterDevice();
__host__ void destroyFilterDevice(Kernel d_k);
__host__ void destroyImage3Device(Image3 *d_img);
__host__ Image3 * makeImage3Device(uint32_t width, uint32_t height);
__host__ void destroyPyramidDevice(Pyramid d_pyr, uint8_t h_nLevels);
__host__ void copyImg3Host2Device(Image3 *d_imgDst, Image3 *h_imgSrc);
__host__ void copyImg3Device2Host(Image3 *h_imgDst, Image3 *d_imgSrc);
__host__ Image3 * getImageFromPyramidDevice(Pyramid d_pyr, uint8_t h_level);
__host__ Pyramid createPyramidDevice(uint32_t width, uint32_t height, uint8_t nLevels);
__host__ void getPyramidDimensionsAtLayer(Pyramid pyr, uint8_t level, uint32_t *width, uint32_t *height);
__global__ void d_copyPyrLevel(Pyramid dst_pyr, Pyramid src_pyr, uint8_t level);
__global__ void d_clampImage3(Image3 *img);
__device__ void d_destroyImage3(Image3 *img);
__device__ void d_imgcpy3(Image3 *d_dest, Image3 *d_source);
__device__ float d_smoothstep(float a, float b, float u);
__device__ float d_clamp(float a, float min_, float max_);
__device__ void d_destroydPyramid(Pyramid pyr, uint8_t nLevels);
__device__ Image3 * d_makeImage3(uint32_t width, uint32_t height);
__device__ Pyramid d_createPyramid(uint32_t width, uint32_t height, uint8_t nLevels);
__device__ void d_remap(Image3 * img, const Pixel3 g0, float sigma, float alpha, float beta);
__device__ inline Pixel3 d_remapSinglePixel(const Pixel3 source, const Pixel3 g0, float sigma, float alpha, float beta);
__device__ void d_subimage3(Image3 *dest, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY);
__device__ void d_subimage3Remap(Image3 *dest, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY, const Pixel3 g0, float sigma, float alpha, float beta);