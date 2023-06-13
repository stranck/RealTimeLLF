#include "processingHandler.h"

#if CUDA_VERSION
	#include "../CUDA/cuda.cuh"
#elif OPENMP_VERSION
	#include "../OpenMP/openmp.h"
#else
	#include "../llf/llf.h"
#endif

volatile bool working = false; //false when we should exit the working loop
volatile uint32_t widthIn = 1, heightIn = 1, widthOut = 1, heightOut = 1; //in and out dimensions of the frames
uint64_t renderingBufferDimension = 0; //dimension in pixels of the buffers used by the rendering part
uint64_t ioBufferDimension = 0; //dimension in pixels of the mt2pt and pt2mt buffers
WorkingBuffers *workingBuffers; //preallocated working buffers used by the selected renderer
Semaphore ioBufferSemaphore; //mutex to acces the IO buffers
Semaphore frameAvailable; //semaphore used to notify the processing thread that a new frame is available
Semaphore cleanupDone; //semaphore used to notify the main thread that the processing thread is done cleaning the buffers and it's shutting down
Pixel4u8 *hostMT2PTbuffer; //mainThread -> processingThread buffer
Pixel4u8 *hostPT2MTbuffer; //processingThread -> mainThread buffer
Image3 *workingImage; //preallocated image that will be processed by the selected renderer

#ifdef SHOW_TIME_STATS_NDI
	TimeData timeData;
	TimeCounter passed = 0;
	uint32_t framesCount = 0;
#endif

//llf parameters
uint8_t _nLevels;
float _sigma, _alpha, _beta;
//extra parameters for the openmp and cuda implementation
#if CUDA_VERSION || OPENMP_VERSION
	uint32_t _nThreads;
	#if CUDA_VERSION
		uint32_t _nBlocks;
	#endif
#endif

//Thread entry point for both linux and windows version
#ifdef ON_WINDOWS
	LPDWORD processingTID;
	DWORD WINAPI processingThread_entryPoint(LPVOID lpParameter){ processingThread(); return 0; }
#else
	pthread_t processingTID;
	void * processingThread_entryPoint(void *param){ processingThread(); return NULL; }
#endif


#if CUDA_VERSION
	/**
	 * @brief Sets the parameters of the processing thread and starts it
	 * 
	 * @param sigma Treshold used by remap function to identify edges and details
	 * @param alpha Controls the details level
	 * @param beta Controls the tone mapping level
	 * @param nLevels Number of layers of the llf's pyramids 
	 * @param nThreads Number of threads for each block executing the llf rendering
	 * @param nBlocks Number of blocks executing the llf rendering
	 */
	void startProcessingThread(float sigma, float alpha, float beta, uint8_t nLevels, uint32_t nThreads, uint32_t nBlocks){
#elif OPENMP_VERSION
	/**
	 * @brief Sets the parameters of the processing thread and starts it
	 * 
	 * @param sigma Treshold used by remap function to identify edges and details
	 * @param alpha Controls the details level
	 * @param beta Controls the tone mapping level
	 * @param nLevels Number of layers of the llf's pyramids 
	 * @param nThreads Number of cpu threads executing the llf rendering
	 */
	void startProcessingThread(float sigma, float alpha, float beta, uint8_t nLevels, uint32_t nThreads){
#else
	/**
	 * @brief Sets the parameters of the processing thread and starts it
	 * 
	 * @param sigma Treshold used by remap function to identify edges and details
	 * @param alpha Controls the details level
	 * @param beta Controls the tone mapping level
	 * @param nLevels Number of layers of the llf's pyramids 
	 */
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
 * @brief Copy the just-received ndi frame into the mainThread->processingThread buffer, so it will be used in the next processing thread iteration
 * 
 * @param ndiVideoFrame Frame just received
 */
void handleIncomingFrame(NDIlib_video_frame_v2_t *ndiVideoFrame){
	ioBufferSemaphore.acquire(); //Acquire the semaphore to access the mainThread<->processingThread buffers
	widthIn = ndiVideoFrame->xres;
	heightIn = ndiVideoFrame->yres; //Updates the size of the last received frame
	uint64_t frameDimension = widthIn * heightIn;
	uint64_t frameDimensionBytes = frameDimension * sizeof(Pixel4u8);
	//Only if the new frame is bigger than the previous frames, increase the buffers size, so we don't reduce the size before we output the rendered frame
	if(frameDimension > ioBufferDimension){
		free(hostMT2PTbuffer);
		free(hostPT2MTbuffer);
		hostMT2PTbuffer = (Pixel4u8 *) malloc(frameDimensionBytes);
		hostPT2MTbuffer = (Pixel4u8 *) malloc(frameDimensionBytes);
		ioBufferDimension = frameDimension;
	}
	//Copy the frame data onto the mainThread->processingThread buffer
	memcpy(hostMT2PTbuffer, ndiVideoFrame->p_data, frameDimensionBytes);
	ioBufferSemaphore.release(); //Release the semaphore to access the mainThread<->processingThread buffers
	frameAvailable.release(); //Notify the processing thread that a new frame is available
}
/**
 * @brief Obtains the last rendered frame and copies it into the ndiVideoFrame
 * 
 * @param ndiVideoFrame Frame to be sent out
 */
void getOutputFrame(NDIlib_video_frame_v2_t *ndiVideoFrame){
	ioBufferSemaphore.acquire(); //Acquire the semaphore to access the mainThread<->processingThread buffers
	uint64_t frameDimensionBytes = ioBufferDimension * sizeof(Pixel4u8);
	ndiVideoFrame->xres = llf_min(ndiVideoFrame->xres, widthOut);
	ndiVideoFrame->yres = llf_min(ndiVideoFrame->yres, heightOut); //Copies back the dimensions of the frame
	uint64_t outFrameDim = ndiVideoFrame->xres * ndiVideoFrame->yres * sizeof(Pixel4u8);
	frameDimensionBytes = llf_min(frameDimensionBytes, outFrameDim); 
	memcpy(ndiVideoFrame->p_data, hostPT2MTbuffer, frameDimensionBytes); //Copies min(lastRenderedFrameDimensions, ndiVideoFrameDimensions) pixels from the processingThread->mainThread buffer to the ndiVideoFrame
	ioBufferSemaphore.release(); //Releases the semaphore to access the mainThread<->processingThread buffers
}

/**
 * @brief Kills the processing thread and makes it release the buffers. It also waits that it has finished doing its stuff
 */
void destroyProcessingThread(){
	if(working){
		working = false;
		print("Waiting for processing thread to finish loop");
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
	ioBufferSemaphore.release();

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
		ioBufferSemaphore.acquire(); //Acquire the semaphore to access the mainThread<->processingThread buffers
		workingImage->width = widthIn;
		workingImage->height = heightIn; //Copy the frame dimensions on the local image
		uint32_t dim = widthIn * heightIn;
		if(dim > renderingBufferDimension){ //If the last frame's dimensions are bigger than previous frames, reallocate the local image and the workingBuffers
			#if OPENMP_VERSION
				destroyWorkingBuffers(workingBuffers, _nLevels, _nThreads);
				initWorkingBuffers(workingBuffers, widthIn, heightIn, _nLevels, _nThreads);
			#else
				destroyWorkingBuffers(workingBuffers, _nLevels);
				initWorkingBuffers(workingBuffers, widthIn, heightIn, _nLevels);
			#endif
			destroyImage3(&workingImage);
			workingImage = makeImage3(widthIn, heightIn);
			renderingBufferDimension = dim;
		}
		Pixel3 *pxs = workingImage->pixels; //Translates the pixels' values from uint8_t to float and copies them into our local image
		for(uint32_t i = 0; i < dim; i++){
			pxs[i].x = hostMT2PTbuffer[i].x / 255.0f;
			pxs[i].y = hostMT2PTbuffer[i].y / 255.0f;
			pxs[i].z = hostMT2PTbuffer[i].z / 255.0f;
		}
		ioBufferSemaphore.release(); //Release the semaphore to access the mainThread<->processingThread buffers
	
		startTimerCounter(timeData);
		//Run the llf rendering function
		#if CUDA_VERSION
			llf(workingImage, _sigma, _alpha, _beta, _nLevels, _nThreads, _nBlocks, workingBuffers);
		#elif OPENMP_VERSION
			llf(workingImage, _sigma, _alpha, _beta, _nLevels, _nThreads, workingBuffers);
		#else
			llf(workingImage, _sigma, _alpha, _beta, _nLevels, workingBuffers);
		#endif
		stopTimerCounter(timeData, passed);

		#ifdef SHOW_TIME_STATS_NDI
			if(++framesCount >= PRINT_STAT_EVERY_N_FRAMES){
				framesCount = 0;
				float time = ((float) ((uint32_t) passed)) / ((float) PRINT_STAT_EVERY_N_FRAMES);
				passed = 0;
				printff("Avarage rendering time of the last %d frames: %.3fms\n", PRINT_STAT_EVERY_N_FRAMES, time);
			}
		#endif

		ioBufferSemaphore.acquire(); //Acquire the semaphore to access the mainThread<->processingThread buffers
		widthOut = workingImage->width;
		heightOut = workingImage->height; //Updates the output dimension
		for(uint32_t i = 0; i < dim; i++){ //Translate the pixels' values from float to uint8_t and copies them into the processingThread->mainThread buffer
			hostPT2MTbuffer[i].x = roundfu8(255.0f * pxs[i].x);
			hostPT2MTbuffer[i].y = roundfu8(255.0f * pxs[i].y);
			hostPT2MTbuffer[i].z = roundfu8(255.0f * pxs[i].z);
			hostPT2MTbuffer[i].w = 0xff;
		}
		ioBufferSemaphore.release(); //Release the semaphore to access the mainThread<->processingThread buffers
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
	cleanupDone.release(); //Notify the mainThread we're done cleaning up everything
	print("Processing thread is shutting down");
}