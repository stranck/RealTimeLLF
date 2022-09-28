#include "openmp.h"

/**
 * @brief Gets the rendered pixel that can directly be placed into the output laplacian inside the llf's main loop
 * 
 * This is a major optimization over the "normal" algorithm:
 * By the paper we have to build the full laplacian pyramid, just to get one pixel from the second lowest layer.
 * Since all layers of a laplacian pyramid are independent from each other, we can just upsample the latest layer of the source gauss pyramid,
 * take the correct pixel and subtract it from the correct pixel of the 2nd latest layer of the gauss pyramid.
 * And not only that! Since the pixel we're taking from the upsampled image depends only by the sorrounding pixels (because we're applying a blur kernel),
 * we can just upsample and convolve the small area of the latest gauss pyramid layer that influences the pixel we're using in the subtraction 
 * 
 * @param source image we're going to upsample. This should be the latest layer of the gauss pyramid
 * @param gaussPx the correct pixel of the current gauss pyramid we're gonna use in the subtraction
 * @param kernel blur kernel
 * @param i x coordinates of the pixel on the upsampled image we're gonna use in the subtraction
 * @param j y coordinates of the pixel on the upsampled image we're gonna use in the subtraction
 * @return Pixel3 Single pixel we've rendered instead of the complete pyramid, we can place directly on the output laplacian pyramid
 */
Pixel3 upsampleConvolveSubtractSinglePixel(Image3 *source, Pixel3 *gaussPx, Kernel kernel, uint32_t i, uint32_t j){
	//This function will only work on the interested subregion get the pixel we're gonna place in the final laplacian pyramid
	uint32_t smallWidth = source->width, smallHeight = source->height;
	Pixel3* srcPx = source->pixels;
	const int32_t  xstart = -1 * KERNEL_DIMENSION / 2;
	const int32_t  ystart = -1 * KERNEL_DIMENSION / 2;
	
	Pixel3 ups = zero3vect;
	for (uint32_t y = 0; y < KERNEL_DIMENSION; y++) {
		int32_t jy = (j + ystart + y) / 2;
		for (uint32_t x = 0; x < KERNEL_DIMENSION; x++) { //For every pixel sorrounding in a KERNEL_DIMENSION^2 square the pixel we're interested located at {i; j}
			int32_t ix = (i + xstart + x) / 2; //Half the coordinate to use them on the original smaller image
			if (ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight) { //If we're in bounds of the smaller image
				float kern_elem = kernel[getKernelPosition(x, y)]; //Take the kernel element
				Pixel3 px = *getPixel3(source, ix, jy); //Take the pixel from the smaller image

				ups.x += px.x * kern_elem;
				ups.y += px.y * kern_elem;
				ups.z += px.z * kern_elem; //Apply the kernel element
			} else {
				float kern_elem = kernel[getKernelPosition(x, y)];
				Pixel3 px = *getPixel3(source, i / 2, j / 2);

				ups.x += px.x * kern_elem;
				ups.y += px.y * kern_elem;
				ups.z += px.z * kern_elem; //If we're out of bounds we will take or 0, or the image limit
			}
		}
	}
	vec3Sub(ups, *gaussPx, ups); //Subtract the blurred pixel we just made from the corresponding pixel of the gaussian pyramid
	return ups;
}
/**
 * @brief Upsamples an image using multiple threads by duplicating it in size and the applying a blur kernel to remove the squares at the same time
 * 
 * This will save an extra copy of the whole upsized image and the need of an extra temp buffer
 * 
 * Unlike upsampleConvolve this function is multithread; each thread gets a batch of pixel to work with, without any dependence from the other threads 
 * 
 * @param dest Destination image
 * @param source Source image 
 * @param kernel Blur kernel
 * @param nThreads Number of parallel threads that will upsample+convolve the image
 */
void upsampleConvolve_parallel(Image3 *dest, Image3 *source, Kernel kernel, const uint8_t nThreads){
	const uint32_t smallWidth = source->width, smallHeight = source->height;
	const uint32_t uppedW = smallWidth << 1;
	const uint32_t uppedH = smallHeight << 1;
	dest->width = uppedW;
	dest->height = uppedH;
	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;
	const uint32_t dim = uppedH * uppedW;

	#pragma omp parallel for num_threads(nThreads) schedule(static) //Each thread is going to handle a different "random" pixel, since there are no dependences between them
	for (int32_t idx = 0; idx < dim; idx++) {
		uint32_t i = idx % uppedW, j = idx / uppedW; //For each pixel C of the image

		Pixel3 c = zero3vect;
		for (uint32_t y = 0; y < rows; y++) {
			int32_t jy = (j + ystart + y) / 2;
			for (uint32_t x = 0; x < cols; x++) { //For each pixel of the kernel square surrounding C 
				int32_t ix = (i + xstart + x) / 2; //Half the coordinate to use them on the original smaller image
				if (ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight) { //If we're in bounds of the smaller image
					float kern_elem = kernel[getKernelPosition(x, y)]; //Take the kernel element
					Pixel3 px = *getPixel3(source, ix, jy); //Take the pixel from the smaller image

					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem; //Apply the kernel element
				} else {
					float kern_elem = kernel[getKernelPosition(x, y)];
					Pixel3 px = *getPixel3(source, i / 2, j / 2);

					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem; //If we're out of bounds we will take or 0, or the image limit
				}
			}
		}
		setPixel3(dest, i, j, &c); //Apply the blurred pixel C to the upsized image
	}
}

/**
 * @brief Collapses a laplacian pyramid reconstructing an image using multithreaded upsampleConvolve
 * 
 * The collapse operations starts from the smallest layer (the one with the greatest index) and proceeds as follows:
 * lapl[n - 1] = lapl[n - 1] + upsample(lapl[n])
 * 
 * As already said, this function is multithread. It uses upsampleConvolve_parallel instead of the normal upsampleConvolve
 * and all add operations are done mulithreaded
 * 
 * @param dest destination image
 * @param laplacianPyr source laplacian pyramid
 * @param nLevels number of layers of the laplacian pyramid
 * @param filter blur kernel for the upsample
 * @param nThreads Number of parallel threads that will be used in the upsampleConvolve and add operations
 */
void collapse(Image3 *dest, Pyramid laplacianPyr, uint8_t nLevels, Kernel filter, const uint8_t nThreads){
	Pixel3 *destPxs = dest->pixels;//We're gonna use the dest image as a temp buffer
	for(int8_t lev = nLevels; lev > 1; lev--){ //For each layer except the last one
		Image3 *currentLevel = laplacianPyr[lev], *biggerLevel = laplacianPyr[lev - 1];
		Pixel3 *biggerLevelPxs = biggerLevel->pixels;

		upsampleConvolve_parallel(dest, currentLevel, filter, nThreads); //Upsample the current lapl layer and temp save it inside the dest image
		uint32_t sizeUpsampled = llf_min(dest->width, biggerLevel->width) * llf_min(dest->height, biggerLevel->height);
		#pragma omp parallel for num_threads(nThreads) schedule(static, 8) //each thread statically gets a batch of 8 pixel to add
		for(int32_t px = 0; px < sizeUpsampled; px++) //For every pixel
			vec3Add(biggerLevelPxs[px], destPxs[px], biggerLevelPxs[px]); //Add them together and save them inside the bigger lapl layer
		biggerLevel->width = dest->width;
		biggerLevel->height = dest->height; //This could cause disalignment problem
	}
	//Handle the last layer separately to save one extra copy
	Image3 *currentLevel = laplacianPyr[1], *biggerLevel = laplacianPyr[0];
	Pixel3 *biggerLevelPxs = biggerLevel->pixels;

	upsampleConvolve_parallel(dest, currentLevel, filter, nThreads);
	uint32_t sizeUpsampled = llf_min(dest->width, biggerLevel->width) * llf_min(dest->height, biggerLevel->height);
	#pragma omp parallel for num_threads(nThreads) schedule(static, 8)
	for(int32_t px = 0; px < sizeUpsampled; px++)
		vec3Add(destPxs[px], destPxs[px], biggerLevelPxs[px]);
}

/**
 * @brief Downsamples an image using multiple threads by halfing it in size and the applying a blur kernel to remove the gaps at the same time
 * 
 * This will save an extra copy of the whole downsized image and the need of an extra temp buffer
 * 
 * Unlike downsampleConvolve this function is multithread; each thread gets a batch of pixel to work with, without any dependence from the other threads
 * 
 * @param dest destination bigger image
 * @param source source smaller image
 * @param width pointer to the width of the source image
 * @param height pointer to the height of the source image
 * @param filter blur kernel
 * @param nThreads Number of parallel threads that will downsample+convolve the image
 */
void downsampleConvolve_parallel(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter, const uint8_t nThreads){
	uint32_t originalW = *width, originalH = *height;
	*width /= 2;
	*height /= 2;
	dest->width = *width;
	dest->height = *height; //Half the image dimension and save both of them in the original ptrs and inside the dest image
	const int32_t startingX = originalW & 1;
	const int32_t startingY = originalH & 1; //If the dimension is odd, we copy only the "middle" pixels. Eg the X: -X-X-
	const int8_t  rows = KERNEL_DIMENSION;
	const int8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;
	originalW -= startingX;
	const uint32_t dim = (originalH - startingY * 2) * originalW; //not *2 on w because we need one extra pixel at the end of the line for the +=2 to work

	#pragma omp parallel for num_threads(nThreads) schedule(static) //Each thread is going to handle a different "random" pixel, since there are no dependences between them
	for(int32_t idx = 0; idx < dim; idx += 2){ //For half of the pixels C
		uint32_t i = (idx % originalW) + startingX, j = (idx / originalW) + startingY;

		Pixel3 c = zero3vect;
		for (uint32_t y = 0; y < rows; y++) {
			int32_t jy = j + (ystart + y) * 2 - startingY;
			for (uint32_t x = 0; x < cols; x++) { //For each pixel in a KERNEL_DIMENSION^2 square sorrounding C
				int32_t ix = i + (xstart + x) * 2 - startingX;

				if (ix >= 0 && ix < originalW && jy >= 0 && jy < originalH) { //If we're in bounds of the bigger image
					float kern_elem = filter[getKernelPosition(x, y)];
					Pixel3 px = *getPixel3(source, ix, jy); //Take the pixel from the bigger image

					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem; //Apply the kernel element
				} else {
					
					float kern_elem = filter[getKernelPosition(x, y)];
					Pixel3 px = *getPixel3(source, i - startingX, j - startingY);

					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem; //If we're out of bounds we will take or 0, or the image limit
				}
			}
		}
		setPixel3(dest, i / 2, j / 2, &c); //Apply the blurred pixel C to the downsized image
	}
}

/**
 * @brief Creates a gaussian pyramid starting from a source image
 * 
 * Each single layer of a gaussian pyramid is defined as follows:
 * gauss[0] = sourceImg
 * gauss[n] = downsample(gauss[n - 1])
 * 
 * Unlike gaussianPyramid, this function saves one extra copy by assuming that in outPyr[0] it's already present the source image
 * 
 * @param outPyr output gaussian pyramid. The layer 0 must have a copy of the source image
 * @param nLevels number of layers of the pyramid
 * @param filter blur kernel
 */
void gaussianPyramid_fast(Pyramid outPyr, uint8_t nLevels, Kernel filter){
	//This function assumes that in outPyr[0] there's already the original image copied
	uint32_t width = outPyr[0]->width, height = outPyr[0]->height;
	for(uint8_t i = 0; i < nLevels; i++)
		downsampleConvolve(outPyr[i + 1], outPyr[i], &width, &height, filter); //Downsample the current layer and save it into the next one
}
/**
 * @brief Creates a gaussian pyramid starting from a source image using multiple threads
 * 
 * Each single layer of a gaussian pyramid is defined as follows:
 * gauss[0] = sourceImg
 * gauss[n] = downsample(gauss[n - 1])
 * 
 * As already said, this function is multithread. It uses downsampleConvolve_parallel and imgcpy3_parallel instead of the normal verions
 * 
 * @param outPyr output gaussian pyramid
 * @param inImg source image
 * @param nLevels number of layers of the pyramid
 * @param filter blur kernel
 */
void gaussianPyramid_parallel(Pyramid outPyr, Image3 *inImg, uint8_t nLevels, Kernel filter, const uint8_t nThreads){
	imgcpy3_parallel(outPyr[0], inImg, nThreads); //parallel copy the first layer
	uint32_t width = inImg->width, height = inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve_parallel(outPyr[1], inImg, &width, &height, filter, nThreads);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve_parallel(outPyr[i + 1], outPyr[i], &width, &height, filter, nThreads); //Downsample the current layer and save it into the next one
}

/**
 * @brief Apply the local laplacian filter over one image using multiple threads
 * 
 * Each thread is in charge of working on a batch of pixels taken from any layer of the gaussian pyramid
 * The rendering of each single pixel it's still "single core", like the normal llf is. We simply parallelize
 * the number of pixel that will be rendered at the same time.
 * We also parallelize the creation of the first gaussian pyramid and the collapse function
 * 
 * Local laplacian filter works as follows:
 * - Create a gaussian pyramid starting from the source image
 * - For each pixel, for each layer of the gauss pyramid:
 * -- take the current pixel G0 from the original gaussian pyramid
 * -- cut a subregion R0 from the source image with a dimension proportional to the layer's dimension near the pixel
 * -- apply a remap function to R0 using G0 as reference
 * -- create a gaussian pyramid over this subregion
 * -- get the pixel GAUSSPX at the correct coordinates respect to the original pixel from the second-last layer of the gaussian pyramid we've just computed
 * -- instead of creating a whole laplacian pyramid, render only the pixel placed at the same coordinates of GAUSSPX, using GAUSSPX and the last layer of the gaussian pyramid we've just computed
 * -- copy the pixel we've just rendered to the current layer of the output laplacian pyramid
 * - copy the smallest layer of the gaussian pyramid over the output laplacian pyramid
 * - collapse the output laplacian pyramid over the destination image
 * - clamp the destination image
 * 
 * @param img source AND destination image. The content of these image are going to be overwritten after the algorithm completes!
 * @param sigma Treshold used by remap function to identify edges and details
 * @param alpha Controls the details level
 * @param beta Controls the tone mapping level
 * @param nLevels Number of layers of the pyramids
 * @param workingBuffers Pre-allocated data structures that will be used during the processing
 */
void llf(Image3 *img, float sigma, float alpha, float beta, uint8_t nLevels, const uint8_t nThreads, WorkingBuffers *workingBuffers){
	uint32_t width = img->width, height = img->height;
	nLevels = llf_min(nLevels, 5); //Clamps the number of levels
	nLevels = llf_max(nLevels, 3);//int(ceil(std::abs(std::log2(llf_min(width, height)) - 3))) + 2;
	Kernel filter = workingBuffers->filter;

	Pyramid gaussPyramid = workingBuffers->gaussPyramid;
	Pyramid outputLaplacian = workingBuffers->outputLaplacian;

	TimeData timeData;
	TimeCounter passed = 0;

	//print("Creating first gauss pyramid");
	startTimerCounter(timeData);
	gaussianPyramid_parallel(gaussPyramid, img, nLevels, filter, nThreads); //Creates a gaussian pyramid starting from the source image
	stopTimerCounter(timeData, passed);
	//print("Entering main loop");
	// Sadly, due to approxxximation in the downsample function, I can't use sum to calculate the pyramid dimension :(
	//uint32_t t = (0b100 << (nLevels * 2));
	//uint32_t end = (img->width * img->height * ((t - 1) / 3)) / (t / 4); //sum[i=0, n] D / 4^i
	uint32_t *pyrDimensions = workingBuffers->pyrDimensions;

	Pyramid *bArr = workingBuffers->bArr;
	CurrentLevelInfo *cliArr = workingBuffers->cliArr;
	//initialize the infos used by each single thread to remember in which pyramid level are we and other cached data
	#pragma omp parallel num_threads(nThreads)
	{
		initLevelInfo(&(cliArr[getThreadId()]), pyrDimensions, gaussPyramid);
	}

	startTimerCounter(timeData);
	#pragma omp parallel for num_threads(nThreads) schedule(dynamic)
	for(int32_t idx = 0; idx < workingBuffers->end; idx++){ //For each single pixel of the whole gaussian pyramid
		//Obtain the buffers that this thread is gonna use
		int threadId = getThreadId();
		CurrentLevelInfo *cli = &(cliArr[threadId]);
		Pyramid bufferGaussPyramid = workingBuffers->bArr[threadId];

		if(idx >= cli->nextLevelDimension) //If we're done doing the current level of the gaussian pyramid
			updateLevelInfo(cli, pyrDimensions, gaussPyramid); //pass to the next level and load its info
		int32_t localIdx = idx - cli->prevLevelDimension; //Get the index inside the current gauss level

		//Obtain more working buffers and translate the index to x;y coordinates of the current layer
		uint8_t lev = cli->lev;
		Image3 *currentGaussLevel = cli->currentGaussLevel;
		uint32_t gaussianWidth = cli->width;
		uint32_t subregionDimension = cli->subregionDimension;
		uint32_t x = localIdx % gaussianWidth, y = localIdx / gaussianWidth;
		
		//if we changed layer update the cached y data
		if(y != cli->oldY){
			uint32_t full_res_y = (1 << lev) * y;
			uint32_t roi_y1 = full_res_y + subregionDimension + 1;
			cli->base_y = subregionDimension > full_res_y ? 0 : full_res_y - subregionDimension;
			cli->end_y = llf_min(roi_y1, height);
			uint32_t full_res_roi_y = full_res_y - cli->base_y;
			cli->full_res_roi_yShifted = full_res_roi_y >> lev;
			cli->oldY = y;
		}

		uint32_t full_res_x = (1 << lev) * x;
		uint32_t roi_x1 = full_res_x + subregionDimension + 1;
		uint32_t base_x = subregionDimension > full_res_x ? 0 : full_res_x - subregionDimension;
		uint32_t end_x = llf_min(roi_x1, width);
		uint32_t full_res_roi_x = full_res_x - base_x;
		uint32_t full_res_roi_xShifted = full_res_roi_x >> lev;

		Pixel3 g0 = *getPixel3(currentGaussLevel, x, y);  //Get the pixel from the current gaussian level
		subimage3(bufferGaussPyramid[0], img, base_x, end_x, cli->base_y, cli->end_y); //Cut a subregion from the source image using bufferLaplacianPyramid[0] as temp buffer
		remap(bufferGaussPyramid[0], g0, sigma, alpha, beta); //Remap the subregion using g0
		uint8_t currentNLevels = cli->currentNLevels; //Pyramids has lev+1 layers!
		gaussianPyramid_fast(bufferGaussPyramid, currentNLevels, filter); //Build a gauss pyramid over the subregion
		Pixel3 *gausPx = getPixel3(bufferGaussPyramid[currentNLevels - 1], full_res_roi_xShifted, cli->full_res_roi_yShifted); //Obtains the reference pixel from the gauss pyramid we're gonna place on the output laplacian pyramid
		Pixel3 outPx = upsampleConvolveSubtractSinglePixel(bufferGaussPyramid[currentNLevels], gausPx, filter, full_res_roi_xShifted, cli->full_res_roi_yShifted); //Builds a subregion of the last 2 layers of the laplacian pyramid just to merge the previous pixel

		setPixel3(outputLaplacian[lev], x, y, &outPx); //store the rendered pixel on the output laplacian pyramid
	}

	imgcpy3_parallel(outputLaplacian[nLevels], gaussPyramid[nLevels], nThreads); //Parallel copy of the smallest layer of the gauss pyramid over the output laplacian pyramid
	//print("Collapsing");
	collapse(img, outputLaplacian, nLevels, filter, nThreads); //Collapse the output laplacian pyramid over the dest image
	stopTimerCounter(timeData, passed);
	#ifdef SHOW_TIME_STATS
		#if ON_WINDOWS
			printff("Total time: %dms\n", passed);
		#else
			printff("Total time: %lums\n", passed);
		#endif
	#endif

	clampImage3_parallel(img, nThreads); //Clamp the dest image to put all the pixel in the [0;1] bounds
}

/**
 * @brief allocates the data structures needed for llf's multithread processing
 * Each thread will need it's own set of buffer gaussian pyramid to render each single pixel
 * We don't need a laplacian pyramid for each thread since we're using upsampleConvolveSubtractSinglePixel instead of building an actual pyramid
 * 
 * @param workingBuffers non-allocated data structures
 * @param width width of the pyramids
 * @param height height of the pyramids
 * @param nLevels number of layers of the pyramids
 * @param nThreads number of threads that are going to do the llf rendering
 */
void initWorkingBuffers(WorkingBuffers *workingBuffers, uint32_t width, uint32_t height, uint8_t nLevels, uint8_t nThreads){
	workingBuffers->outputLaplacian = createPyramid(width, height, nLevels);
	workingBuffers->gaussPyramid = createPyramid(width, height, nLevels);
	workingBuffers->pyrDimensions = (uint32_t *) malloc((nLevels + 1) * sizeof(uint32_t));
	workingBuffers->end = 0;
	for(uint8_t i = 0; i < nLevels; i++){ //Creates the array with the cached pyramid dimensions for each layer
		Image3 *lev = workingBuffers->gaussPyramid[i];
		uint32_t dim = lev->width * lev->height;
		workingBuffers->pyrDimensions[i] = dim;
		workingBuffers->end += dim;
	}
	workingBuffers->pyrDimensions[nLevels] = workingBuffers->gaussPyramid[nLevels]->width * workingBuffers->gaussPyramid[nLevels]->height;
	workingBuffers->cliArr = (CurrentLevelInfo *) malloc(nThreads * sizeof(CurrentLevelInfo));
	workingBuffers->bArr = (Pyramid *) malloc(nThreads * sizeof(Pyramid));
	for(uint32_t i = 0; i < nThreads; i++) //Creates a buffer gaussian pyramid for each single layer
		workingBuffers->bArr[i] = createPyramid(width, height, nLevels);
	workingBuffers->filter = createFilter();
}
/**
 * @brief destroy the data structures needed for llf's multithread processing
 * 
 * @param workingBuffers allocated data structures
 * @param nLevels number of layers of the already allocated pyramids
 * @param nThreads number of threads used during the llf's multithreaded processing
 */
void destroyWorkingBuffers(WorkingBuffers *workingBuffers, uint8_t nLevels, uint8_t nThreads){
	destroyPyramid(&(workingBuffers->outputLaplacian), nLevels);
	destroyPyramid(&(workingBuffers->gaussPyramid), nLevels);
	for(uint32_t i = 0; i < nThreads; i++)
		destroyPyramid(&(workingBuffers->bArr[i]), nLevels);
	destroyFilter(&(workingBuffers->filter));
	free(workingBuffers->pyrDimensions);
	free(workingBuffers->cliArr);
	free(workingBuffers->bArr);
}