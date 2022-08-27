#pragma once

#ifndef __CUDA_ARCH__
	#define __CUDA_ARCH__ -1
#endif

#define CUDA_INCLUDE __CUDACC__ //had to do this to fix vs code intellisense doing random stuff
//#define CUDART_VERSION 6000 //Sync primitives are only supported with sm_70 arch... I have a gtx1080 which is sm_60. Sad :(
#define SYNC_PRIMITIVES_SUPPORTED 0 //__CUDA_ARCH__ >= 700

#define MAX_LAYERS 5
#define MAX_PYR_LAYER 3 * ((1 << (MAX_LAYERS + 2)) - 1)

#include "../utils/imageutils.h"
#include "../utils/extramath.h"
#include "../utils/llfUtils.h"
#include "../utils/structs.h"
#include "../utils/vects.h"
#include "../utils/utils.h"
#include "bufferManager.cuh"
#include "cudaUtils.cuh"
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#ifdef CUDA_INCLUDE
	#include <cuda.h>
	#include <cuda_runtime.h>
#endif

__device__ void upsampleConvolve(Image3 *dest, Image3 *source, Kernel kernel);
__device__ void downsampleConvolve(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter);
__device__ void __gaussianPyramid_internal(Pyramid d_outPyr, Image3 *d_inImg, uint8_t nLevels, Kernel d_filter);
__global__ void gaussianPyramid(Pyramid d_outPyr, Image3 *d_inImg, uint8_t nLevels, Kernel d_filter);
__device__ void laplacianPyramid(Pyramid laplacian, Pyramid tempGauss, uint8_t nLevels, Kernel filter);
__global__ void collapse(Image3 *dest, Pyramid laplacianPyr, uint8_t nLevels, Kernel filter);
__global__ void __d_llf_internal(Pyramid outputLaplacian, Pyramid gaussPyramid, Image3 *img, uint32_t width, uint32_t height, uint8_t lev, uint32_t subregionDimension, Kernel filter, double sigma, double alpha, double beta, PyrBuffer *buffer);
__host__ void llf(Image3 *h_img, double h_sigma, double h_alpha, double h_beta, uint8_t h_nLevels, uint32_t h_nThreads, uint32_t h_elementsNo);
uint32_t getPixelNoPerPyramid(uint8_t nLevels);
int main();