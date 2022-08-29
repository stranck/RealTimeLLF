#include "bufferManager.cuh"

__device__ NodeBuffer * d_aquireBuffer(PyrBuffer *buffer){
	#if SYNC_PRIMITIVES_SUPPORTED
		buffer->managerMutex.acquire();
		if(!buffer->availableBuffers.try_acquire()){ //There's no available buffer
			buffer->managerMutex.release(); //Release the lock
			buffer->availableBuffers.acquire(); //Wait that a buffer is available
			buffer->managerMutex.acquire(); //Reaquire the lock to operate on the list
		}
		//At this point we should have a free buffer and we can safely operate on the list
		NodeBuffer *first = buffer->first;
		buffer->first = first->next;
		buffer->managerMutex.release(); //We successfully obtained a node, we can now release the lock

		return first;
	#else
		return &(buffer[blockIdx.x].el);
	#endif
}

__device__ void d_releaseBuffer(NodeBuffer *node, PyrBuffer *buffer){
	#if SYNC_PRIMITIVES_SUPPORTED
		printf("a");
		buffer->managerMutex.acquire();
		
		node->next = buffer->first;
		buffer->first = node;
		printf("b");
		buffer->availableBuffers.release();
		
		printf("c");
		buffer->managerMutex.release();
		printf(" ");
	#endif
}


__global__ void __d_createBufferDevice__internal(uint32_t elementsNo, uint32_t pixelSize, uint8_t nLevels, PyrBuffer *buff){
	#if SYNC_PRIMITIVES_SUPPORTED
 
		buff->first = NULL;
	#endif

	for(uint32_t i = 0; i < elementsNo; i++){
		printf("\n__d_createBufferDevice__internal: Adding pyramids at buffer address 0x%012llx    [% 3d]\n", &buff[i], i);
		printf("-- LAPL:\n");
		Pyramid lapl = d_createPyramid(pixelSize, pixelSize, nLevels);
		printf("-- GAUS:\n");
		Pyramid gauss = d_createPyramid(pixelSize, pixelSize, nLevels);

		#if SYNC_PRIMITIVES_SUPPORTED
			NodeBuffer *node;
			cudaMalloc(&node, sizeof(NodeBuffer));
			node->bufferLaplacianPyramid = lapl;
			node->bufferGaussPyramid = gauss;
			printf("release buffer... ");
			d_releaseBuffer(node, buff);
			printf("Released\n");
		#else
			buff[i].el.bufferGaussPyramid = gauss;
			buff[i].el.bufferLaplacianPyramid = lapl;
		#endif
	}
}
__host__ PyrBuffer * createBufferDevice(uint32_t h_elementsNo, uint32_t h_pixelSize, uint8_t h_nLevels){
	PyrBuffer *d_buff;
	#if SYNC_PRIMITIVES_SUPPORTED
		CHECK(cudaMalloc((void**) &d_buff, sizeof(PyrBuffer)));
	#else
		CHECK(cudaMalloc((void**) &d_buff, h_elementsNo * sizeof(PyrBuffer)));
	#endif
	printff("createBufferDevice: Dimensions %03dx%03d @ %d levels [%d els]    Buffer address: 0x%012llx\n", h_pixelSize, h_pixelSize, h_nLevels, h_elementsNo, d_buff);
	__d_createBufferDevice__internal<<<1, 1>>>(h_elementsNo, h_pixelSize, h_nLevels, d_buff);
	fflush(stdout);
	CHECK(cudaDeviceSynchronize());
	return d_buff;
}

__global__ void __d_destroyBufferDevice__internal(uint32_t elementsNo, uint8_t nLevels, PyrBuffer *buff){
	for(uint32_t i = 0; i < elementsNo; i++){
		#if SYNC_PRIMITIVES_SUPPORTED
			NodeBuffer *node = d_aquireBuffer(buff);
			d_destroydPyramid(node->bufferGaussPyramid, nLevels);
			d_destroydPyramid(node->bufferLaplacianPyramid, nLevels);
			cudaFree(node);
		#else
			d_destroydPyramid(buff[i].el.bufferGaussPyramid, nLevels);
			d_destroydPyramid(buff[i].el.bufferLaplacianPyramid, nLevels);
		#endif
	}
	#if SYNC_PRIMITIVES_SUPPORTED
		buff->managerMutex.~counting_semaphore();
		buff->availableBuffers.~counting_semaphore();
	#endif
}
__host__ void destroyBufferDevice(uint32_t elementsNo, uint8_t h_nLevels, PyrBuffer *d_buff){
	__d_destroyBufferDevice__internal<<<1, 1>>>(elementsNo, h_nLevels, d_buff);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaFree(d_buff));
}