#include "bufferManager.cuh"

__device__ NodeBuffer * d_aquireBuffer(PyrBuffer *buffer){
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
}

__device__ void d_releaseBuffer(NodeBuffer *node, PyrBuffer *buffer){
	buffer->managerMutex.acquire();
	
	node->next = buffer->first;
	buffer->first = node;
	buffer->availableBuffers.release();
	
	buffer->managerMutex.release();
}


__global__ void __d_createBufferDevice__internal(uint32_t elementsNo, uint32_t pixelSize, uint8_t nLevels, PyrBuffer *buff){
	buff->managerMutex.binary_semaphore();
	buff->availableBuffers.counting_semaphore();
	buff->first = NULL;

	for(uint32_t i = 0; i < elementsNo; i++){
		Pyramid lapl = d_createPyramid(pixelSize, pixelSize, nLevels);
		Pyramid gauss = d_createPyramid(pixelSize, pixelSize, nLevels);

		NodeBuffer *node;
		cudaMalloc(&node, sizeof(NodeBuffer));
		node->bufferLaplacianPyramid = lapl;
		node->bufferGaussPyramid = gauss;
		d_releaseBuffer(node, buff);
	}
}
__host__ PyrBuffer * createBufferDevice(uint32_t h_elementsNo, uint32_t h_pixelSize, uint8_t h_nLevels){
	PyrBuffer *d_buff;
	CHECK(cudaMalloc((void**) &d_buff, sizeof(PyrBuffer)));
	__d_createBufferDevice__internal<<<1, 1>>>(h_elementsNo, h_pixelSize, h_nLevels, d_buff);
	CHECK(cudaDeviceSynchronize());
	return d_buff;
}

__global__ void __d_destroyBufferDevice__internal(uint32_t elementsNo, uint8_t nLevels, PyrBuffer *buff){

	for(uint32_t i = 0; i < elementsNo; i++){
		NodeBuffer *node = d_aquireBuffer(buff);
		d_destroydPyramid(node->bufferGaussPyramid, nLevels);
		d_destroydPyramid(node->bufferLaplacianPyramid, nLevels);
		cudaFree(node);
	}
	buff->managerMutex.~binary_semaphore();
	buff->availableBuffers.~counting_semaphore();
}
__host__ void destroyBufferDevice(uint32_t elementsNo, uint8_t h_nLevels, PyrBuffer *d_buff){
	__d_destroyBufferDevice__internal<<<1, 1>>>(elementsNo, h_nLevels, d_buff);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaFree(d_buff));
}