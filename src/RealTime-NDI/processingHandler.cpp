#include "processingHandler.h"

#if CUDA_VERSION
	#include "../CUDA/cuda.cuh"
#elif OPENMP_VERSION
	#include "../OpenMP/openmp.h"
#else
	#include "../llf/llf.h"
#endif

volatile bool working = false;
volatile uint32_t widthIn = 1, heightIn = 1, widthOut = 1, heightOut = 1;
uint64_t lastDeviceBufferDimension = 0;
uint64_t lastHostBufferDimension = 0;
WorkingBuffers *workingBuffers;
Semaphore threadsSemaphore;
Semaphore frameAvailable;
Semaphore cleanupDone;
Pixel4u8 *hostMT2PTbuffer;
Pixel4u8 *hostPT2MTbuffer;
Image3 *workingImage;

float _sigma, _alpha, _beta;
#if CUDA_VERSION || OPENMP_VERSION
	uint32_t _nThreads;
	#if CUDA_VERSION
		uint32_t _nBlocks;
	#endif
#endif
uint8_t _nLevels;

#ifdef ON_WINDOWS
	LPDWORD processingTID;
	DWORD WINAPI processingThread_entryPoint(LPVOID lpParameter){ processingThread(); return 0; }
#else
	pthread_t processingTID;
	void * processingThread_entryPoint(void *param){ processingThread(); return NULL; }
#endif

/**
 * @brief Sets the parameters to the processing thread and starts it
 * 
 * @param sigma Treshold used by remap function to identify edges and details
 * @param alpha Controls the details level
 * @param beta Controls the tone mapping level
 * @param nLevels Number of layers of the llf's pyramids 
 * @param nThreads CUDA: Number of threads for each block executing the llf rendering OPENMP: Number of cpu threads executing the llf rendering
 * @param nBlocks CUDA: Number of blocks executing the llf rendering
 */
#if CUDA_VERSION
	void startProcessingThread(float sigma, float alpha, float beta, uint8_t nLevels, uint32_t nThreads, uint32_t nBlocks){
#elif OPENMP_VERSION
	void startProcessingThread(float sigma, float alpha, float beta, uint8_t nLevels, uint32_t nThreads){
#else
	void startProcessingThread(float sigma, float alpha, float beta, uint8_t nLevels){
#endif
	//Updates the processing paramethers 
	_sigma = sigma;
	_alpha = alpha;
	_beta = beta;
	_nLevels = nLevels;
	#if CUDA_VERSION || OPENMP_VERSION
		_nThreads = nThreads;
		#if CUDA_VERSION
			_nBlocks = nBlocks;
		#endif
	#endif

	//Launch the processing thread
	#ifdef ON_WINDOWS
		CreateThread(NULL, 0, processingThread_entryPoint, NULL, 0, processingTID);
	#else
		pthread_create(&processingTID, 0, processingThread_entryPoint, NULL);
	#endif
}

/**
 * @brief Copy the just-received ndi frame into the mainThread->processingThread buffer, so it will be used at the next processing thread iteration
 * 
 * @param ndiVideoFrame Frame just received
 */
void handleIncomingFrame(NDIlib_video_frame_v2_t *ndiVideoFrame){
	threadsSemaphore.acquire(); //Acquire the semaphore to access the mainThread<->processingThread buffers
	widthIn = ndiVideoFrame->xres;
	heightIn = ndiVideoFrame->yres; //Updates the size of the last received frame
	uint64_t frameDimension = widthIn * heightIn;
	uint64_t frameDimensionBytes = frameDimension * sizeof(Pixel4u8);
	//If the new frame is bigger than the previous frames, increase the buffers size
	if(frameDimension > lastHostBufferDimension){ //so we don't reduce the size before we output the rendered frame
		free(hostMT2PTbuffer);
		free(hostPT2MTbuffer);
		hostMT2PTbuffer = (Pixel4u8 *) malloc(frameDimensionBytes);
		hostPT2MTbuffer = (Pixel4u8 *) malloc(frameDimensionBytes);
		lastHostBufferDimension = frameDimension;
	}
	//Copy the frame data onto the mainThread->processingThread buffer
	memcpy(hostMT2PTbuffer, ndiVideoFrame->p_data, frameDimensionBytes);
	threadsSemaphore.release(); //Release the semaphore to access the mainThread<->processingThread buffers
	frameAvailable.release(); //Notify the processing thread that a new frame is available
}
/**
 * @brief Obtains the last rendered frame and copies it into the ndiVideoFrame
 * 
 * @param ndiVideoFrame Frame to be sent out
 */
void getOutputFrame(NDIlib_video_frame_v2_t *ndiVideoFrame){
	threadsSemaphore.acquire(); //Acquire the semaphore to access the mainThread<->processingThread buffers
	uint64_t frameDimensionBytes = lastHostBufferDimension * sizeof(Pixel4u8);
	ndiVideoFrame->xres = llf_min(ndiVideoFrame->xres, widthOut);
	ndiVideoFrame->yres = llf_min(ndiVideoFrame->yres, heightOut);
	uint64_t outFrameDim = ndiVideoFrame->xres * ndiVideoFrame->yres * sizeof(Pixel4u8);
	frameDimensionBytes = llf_min(frameDimensionBytes, outFrameDim); 
	memcpy(ndiVideoFrame->p_data, hostPT2MTbuffer, frameDimensionBytes); //Copies min(lastRenderedFrameDimensions, ndiVideoFrameDimensions) pixels from the processingThread->mainThread buffer to the ndiVideoFrame
	threadsSemaphore.release(); //Releases the semaphore to access the mainThread<->processingThread buffers
}

/**
 * @brief Kills the processing thread and makes it release the buffers. It also waits that it has finished doing its stuff
 */
void destroyProcessingThread(){
	if(working){
		print("Waiting for processing thread to finish loop");
		working = false;
		frameAvailable.release(); //Release if thread was waiting for a frame
		cleanupDone.acquire(); //wait for thread to finish
	}
}

/**
 * @brief Initializes the buffers used by the processing thread and the llf algorithm to a predefined small size
 */
void initProcessingThread(){
	hostMT2PTbuffer = (Pixel4u8 *) malloc(1);
	hostPT2MTbuffer = (Pixel4u8 *) malloc(1);
	threadsSemaphore.release();

	workingImage = makeImage3(1, 1);

	workingBuffers = (WorkingBuffers *) malloc(sizeof(WorkingBuffers));
	#if OPENMP_VERSION
		initWorkingBuffers(workingBuffers, 200, 200, _nLevels, _nThreads);
	#else
		initWorkingBuffers(workingBuffers, 200, 200, _nLevels);
	#endif
}
/**
 * @brief Processing thread. It works as follows:
 * - init the buffers
 * - while(it's working):
 * -- wait for an available frame
 * -- copy the frame to a local image
 * -- render the image using llf 
 * -- copy back the rendered image to the processingThread->mainThread buffer
 */
void processingThread(){
	initProcessingThread();
	working = true;
	while(working){
		frameAvailable.acquire(); //Wait for an available frame

		//Copies the image locally
		threadsSemaphore.acquire(); //Acquire the semaphore to access the mainThread<->processingThread buffers
		workingImage->width = widthIn;
		workingImage->height = heightIn; //Copy the frame dimensions on the local image
		uint32_t dim = widthIn * heightIn;
		if(dim > lastDeviceBufferDimension){ //If the last frame's dimensions are bigger than previous frames, reallocate the local image and the workingBuffers
			destroyImage3(&workingImage);
			#if OPENMP_VERSION
				destroyWorkingBuffers(workingBuffers, _nLevels, _nThreads);
				initWorkingBuffers(workingBuffers, widthIn, heightIn, _nLevels, _nThreads);
			#else
				destroyWorkingBuffers(workingBuffers, _nLevels);
				initWorkingBuffers(workingBuffers, widthIn, heightIn, _nLevels);
			#endif
			workingImage = makeImage3(widthIn, heightIn);
			lastDeviceBufferDimension = dim;
		}
		Pixel3 *pxs = workingImage->pixels; //Translates the pixels' values from uint8_t to float and copies them into our local image
		for(uint32_t i = 0; i < dim; i++){
			pxs[i].x = hostMT2PTbuffer[i].x / 255.0f;
			pxs[i].y = hostMT2PTbuffer[i].y / 255.0f;
			pxs[i].z = hostMT2PTbuffer[i].z / 255.0f;
		}
		threadsSemaphore.release(); //Release the semaphore to access the mainThread<->processingThread buffers
	
		//Run the llf rendering function
		#if CUDA_VERSION
			llf(workingImage, _sigma, _alpha, _beta, _nLevels, _nThreads, _nBlocks, workingBuffers);
		#elif OPENMP_VERSION
			llf(workingImage, _sigma, _alpha, _beta, _nLevels, _nThreads, workingBuffers);
		#else
			llf(workingImage, _sigma, _alpha, _beta, _nLevels, workingBuffers);
		#endif

		threadsSemaphore.acquire(); //Acquire the semaphore to access the mainThread<->processingThread buffers
		widthOut = workingImage->width;
		heightOut = workingImage->height; //Updates the output dimension
		for(uint32_t i = 0; i < dim; i++){ //Translate the pixels' values from float to uint8_t and copies them into the processingThread->mainThread buffer
			hostPT2MTbuffer[i].x = roundfu8(255.0f * pxs[i].x);
			hostPT2MTbuffer[i].y = roundfu8(255.0f * pxs[i].y);
			hostPT2MTbuffer[i].z = roundfu8(255.0f * pxs[i].z);
			hostPT2MTbuffer[i].w = 0xff;
		}
		threadsSemaphore.release(); //Release the semaphore to access the mainThread<->processingThread buffers
	}

	//Free all the buffers
	print("Processing thread is destroying his stuff");
	free(hostMT2PTbuffer);
	free(hostPT2MTbuffer);
	destroyImage3(&workingImage);
	#if OPENMP_VERSION
		destroyWorkingBuffers(workingBuffers, _nLevels, _nThreads);
	#else
		destroyWorkingBuffers(workingBuffers, _nLevels);
	#endif
	cleanupDone.release(); //Notify the mainThread we're done
	print("Processing thread is done");
}