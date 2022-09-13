#include "cudaHandler.cuh"

#include "../CUDA/cuda.cuh"
#include <cuda/semaphore>

cuda::counting_semaphore<cuda::thread_scope_system, 1> frameAvailable;
cuda::counting_semaphore<cuda::thread_scope_system> hostSemaphore;
volatile uint32_t widthIn = 0, heightIn = 0, widthOut = 0, heightOut = 0;
uint64_t lastDeviceBufferDimension = 0;
uint64_t lastHostBufferDimension = 0;
CUDAbuffers *cudaBuffers;
Pixel4u8 *hostN2Tbuffer;
Pixel4u8 *hostT2Nbuffer;
Image3 *workingImage;

float _sigma, _alpha, _beta;
uint32_t _nThreads, _nBlocks;
uint8_t _nLevels;

#ifdef ON_WINDOWS
	LPDWORD gpuProcessingTID;
	DWORD WINAPI gpuProcessingThread_entryPoint(LPVOID lpParameter){ gpuProcessingThread(); return 0; }
#else
	pthread_t gpuProcessingTID;
	void * gpuProcessingThread_entryPoint(void *param){ gpuProcessingThread(); return NULL; }
#endif
void startGpuProcessingThread(float sigma, float alpha, float beta, uint8_t nLevels, uint32_t nThreads, uint32_t nBlocks){
	_sigma = sigma;
	_alpha = alpha;
	_beta = beta;
	_nLevels = nLevels;
	_nThreads = nThreads;
	_nBlocks = nBlocks;
	#ifdef ON_WINDOWS
		CreateThread(NULL, 0, gpuProcessingThread_entryPoint, NULL, 0, gpuProcessingTID);
	#else
		pthread_create(&gpuProcessingTID, 0, gpuProcessingThread_entryPoint, NULL);
	#endif
}

void handleIncomingFrame(NDIlib_video_frame_v2_t *ndiVideoFrame){
	hostSemaphore.acquire();
	widthIn = ndiVideoFrame->xres;
	heightIn = ndiVideoFrame->yres;
	uint64_t frameDimension = widthIn * heightIn;
	uint64_t frameDimensionBytes = frameDimension * sizeof(Pixel4u8);
	if(frameDimension > lastHostBufferDimension){ //so we don't reduce the size before we output the rendered frame
		free(hostN2Tbuffer);
		free(hostT2Nbuffer);
		hostN2Tbuffer = (Pixel4u8 *) malloc(frameDimensionBytes);
		hostT2Nbuffer = (Pixel4u8 *) malloc(frameDimensionBytes);
		lastHostBufferDimension = frameDimension;
	}
	memcpy(hostN2Tbuffer, ndiVideoFrame->p_data, frameDimensionBytes);
	hostSemaphore.release();
	frameAvailable.release();
}
void writeOutputFrame(NDIlib_video_frame_v2_t *ndiVideoFrame){
	hostSemaphore.acquire();
	uint64_t frameDimensionBytes = lastHostBufferDimension * sizeof(Pixel4u8);
	ndiVideoFrame->xres = min(ndiVideoFrame->xres, widthOut);
	ndiVideoFrame->yres = min(ndiVideoFrame->yres, heightOut);
	uint64_t outFrameDim = widthOut * heightOut * sizeof(Pixel4u8);
	frameDimensionBytes = min(frameDimensionBytes, outFrameDim);
	memcpy(ndiVideoFrame->p_data, hostT2Nbuffer, frameDimensionBytes);
	hostSemaphore.release();
}


void initProcessingThread(){
	hostN2Tbuffer = (Pixel4u8 *) malloc(1);
	hostT2Nbuffer = (Pixel4u8 *) malloc(1);
	hostSemaphore.release();

	workingImage = makeImage3(1, 1);

	cudaBuffers = (CUDAbuffers *) malloc(sizeof(CUDAbuffers));
	initCUDAbuffers(cudaBuffers, 200, 200, _nLevels);
}
void gpuProcessingThread(){
	initProcessingThread();
	while(true){
		frameAvailable.acquire(); //Wait for an available frame

		hostSemaphore.acquire(); //Copies the image locally
		workingImage->width = widthIn;
		workingImage->height = heightIn;
		uint32_t dim = widthIn * heightIn;
		if(dim > lastDeviceBufferDimension){
			destroyImage3(&workingImage);
			destroyCUDAbuffers(cudaBuffers, _nLevels);
			initCUDAbuffers(cudaBuffers, widthIn, heightIn, _nLevels);
			workingImage = makeImage3(widthIn, heightIn);
			lastDeviceBufferDimension = dim;
		}
		Pixel3 *pxs = workingImage->pixels;
		for(uint32_t i = 0; i < dim; i++){
			pxs[i].x = hostN2Tbuffer[i].x / 255.0f;
			pxs[i].y = hostN2Tbuffer[i].y / 255.0f;
			pxs[i].z = hostN2Tbuffer[i].z / 255.0f;
		}
		hostSemaphore.release();
	
		llf(workingImage, _sigma, _alpha, _beta, _nLevels, _nThreads, _nBlocks, cudaBuffers);

		hostSemaphore.acquire();
		widthOut = workingImage->width;
		heightOut = workingImage->height;
		for(uint32_t i = 0; i < dim; i++){
			hostT2Nbuffer[i].x = roundfu8(255.0f * pxs[i].x);
			hostT2Nbuffer[i].y = roundfu8(255.0f * pxs[i].y);
			hostT2Nbuffer[i].z = roundfu8(255.0f * pxs[i].z);
			hostT2Nbuffer[i].w = 0xff;
		}
		hostSemaphore.release();
	}
}