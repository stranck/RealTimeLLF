#pragma once

typedef struct NodeBuffer NodeBuffer;
typedef struct PyrBuffer PyrBuffer;

#include <stdint.h>
#include "../utils/structs.h"
#include "cuda.cuh"

#ifdef CUDA_INCLUDE
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda/semaphore>
#else
	#include <semaphore>
#endif

struct NodeBuffer {
	Pyramid bufferLaplacianPyramid;
	Pyramid bufferGaussPyramid;
	NodeBuffer *next;
};

struct PyrBuffer {
	cuda::binary_semaphore<cuda::thread_scope_device> managerMutex;
	cuda::counting_semaphore<cuda::thread_scope_device> availableBuffers;
	NodeBuffer *first;
};

__device__ NodeBuffer * d_aquireBuffer(PyrBuffer *buffer);
__device__ void d_releaseBuffer(NodeBuffer *node, PyrBuffer *buffer);

__host__ PyrBuffer * createBufferDevice(uint32_t h_elementsNo, uint32_t h_pixelSize, uint8_t h_nLevels);
__host__ void destroyBufferDevice(uint32_t elementsNo, uint8_t h_nLevels, PyrBuffer *d_buff);