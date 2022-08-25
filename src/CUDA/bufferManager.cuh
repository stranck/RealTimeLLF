#pragma once

typedef struct NodeBuffer NodeBuffer;
typedef struct PyrBuffer PyrBuffer;

#include <stdint.h>
#include "../utils/structs.h"
#include "cuda.cuh"

#ifdef CUDA_INCLUDE
	#include <cuda.h>
	#include <cuda_runtime.h>
	#if SYNC_PRIMITIVES_SUPPORTED
		#include <cuda/semaphore>
	#endif
#else
	#include <semaphore>
#endif

struct NodeBuffer {
	Pyramid bufferLaplacianPyramid;
	Pyramid bufferGaussPyramid;
	#if SYNC_PRIMITIVES_SUPPORTED
		NodeBuffer *next;
	#endif
};

struct PyrBuffer {
	#if SYNC_PRIMITIVES_SUPPORTED
		cuda::binary_semaphore<cuda::thread_scope_device> managerMutex;
		cuda::counting_semaphore<cuda::thread_scope_device> availableBuffers;
		NodeBuffer *first;
	#else
		NodeBuffer el;
	#endif
};

__device__ NodeBuffer * d_aquireBuffer(PyrBuffer *buffer);
__device__ void d_releaseBuffer(NodeBuffer *node, PyrBuffer *buffer);

__host__ PyrBuffer * createBufferDevice(uint32_t h_elementsNo, uint32_t h_pixelSize, uint8_t h_nLevels);
__host__ void destroyBufferDevice(uint32_t elementsNo, uint8_t h_nLevels, PyrBuffer *d_buff);