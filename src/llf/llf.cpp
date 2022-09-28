#include "llf.h"

/**
 * @brief Upsamples an image by duplicating it in size and the applying a blur kernel to remove the squares
 * 
 * This is the original and cleanest upsample + convolve version from the llf's paper
 * 
 * @param dest Destination image
 * @param source Source image 
 * @param filter Blur kernel
 * @param buffer Buffer used to first resize the image and then as source to the convolve algorithm
 */
void upsample(Image3 *dest, Image3 *source, Kernel filter, Image3 *buffer){
	uint32_t smallWidth = source->width, smallHeight = source->height;
	uint32_t uppedW = smallWidth << 1;
	uint32_t uppedH = smallHeight << 1;
	buffer->width = uppedW;
	buffer->height = uppedH; //Duplicate the size of source and save it in dest
	for(uint32_t y = 0; y < smallHeight; y++){
		uint32_t yUp = y * 2;
		uint32_t yUpLess = yUp++;
		for(uint32_t x = 0; x < smallWidth; x++){ //For each pixel of the source small image
			uint32_t xUp = x * 2;
			Pixel3 *pixel = getPixel3(source, x, y);
			uint32_t xUpLess = xUp++;

			setPixel3(buffer, xUpLess, yUpLess, pixel); //copy the pixel in a square sorrounding it
			setPixel3(buffer, xUpLess, yUp, pixel);
			setPixel3(buffer, xUp, yUpLess, pixel);
			setPixel3(buffer, xUp, yUp, pixel);
		}
	}
	convolve(dest, buffer, filter); //apply the blur kernel to smooth the squares
}

/**
 * @brief Collapses a laplacian pyramid reconstructing an image
 * 
 * The collapse operations starts from the smallest layer (the one with the greatest index) and proceeds as follows:
 * lapl[n - 1] = lapl[n - 1] + upsample(lapl[n])
 * 
 * @param dest destination image
 * @param laplacianPyr source laplacian pyramid
 * @param nLevels number of layers of the laplacian pyramid
 * @param filter blur kernel for the upsample
 */
void collapse(Image3 *dest, Pyramid laplacianPyr, uint8_t nLevels, Kernel filter){
	Pixel3 *destPxs = dest->pixels; //We're gonna use the dest image as a temp buffer
	for(int8_t lev = nLevels; lev > 1; lev--){ //For each layer except the last one
		Image3 *currentLevel = laplacianPyr[lev], *biggerLevel = laplacianPyr[lev - 1];
		Pixel3 *biggerLevelPxs = biggerLevel->pixels;

		upsampleConvolve(dest, currentLevel, filter); //Upsample the current lapl layer and temp save it inside the dest image
		uint32_t sizeUpsampled = llf_min(dest->width, biggerLevel->width) * llf_min(dest->height, biggerLevel->height);
		for(uint32_t px = 0; px < sizeUpsampled; px++) //For every pixel
			vec3Add(biggerLevelPxs[px], destPxs[px], biggerLevelPxs[px]); //Add them together and save them inside the bigger lapl layer 
		biggerLevel->width = dest->width;
		biggerLevel->height = dest->height; //This could cause disalignment problem
	}
	//Handle the last layer separately to save one extra copy
	Image3 *currentLevel = laplacianPyr[1], *biggerLevel = laplacianPyr[0];
	Pixel3 *biggerLevelPxs = biggerLevel->pixels;

	upsampleConvolve(dest, currentLevel, filter);
	uint32_t sizeUpsampled = llf_min(dest->width, biggerLevel->width) * llf_min(dest->height, biggerLevel->height);
	for(uint32_t px = 0; px < sizeUpsampled; px++)
		vec3Add(destPxs[px], destPxs[px], biggerLevelPxs[px]);
}

/**
 * @brief Downsamples an image by halfing it in size and the applying a blur kernel to remove the gaps
 * 
 * This is the original and cleanest convolve + downsample version from the llf's paper
 * 
 * @param dest destination bigger image
 * @param source source smaller image
 * @param width pointer to the width of the source image
 * @param height pointer to the height of the source image
 * @param filter blur kernel
 * @param buffer Buffer used to first resize the image and then as source to the convolve algorithm
 */
void downsample(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter, Image3 *buffer){
	convolve(buffer, source, filter); //blur the source image and save it into the buffer
	uint32_t originalW = *width, originalH = *height;
	*width /= 2;
	*height /= 2;
	dest->width = *width;
	dest->height = *height; //Half the image dimension and save both of them in the original ptrs and inside the dest image
	uint32_t y;
	uint32_t startingX = originalW & 1;
	uint32_t startingY = originalH & 1; //If the dimension is odd, we copy only the "middle" pixels. Eg the X: -X-X-
	for(y = startingY; y < originalH; y += 2) {
		uint32_t x;
		for(x = startingX; x < originalW; x += 2) {
			setPixel3(dest, x / 2, y / 2, getPixel3(buffer, x - startingX, y - startingY)); //Copy only half of the pixels to the dest image
		}
	}
}

/**
 * @brief Apply the local laplacian filter over one image
 * 
 * This is the cleanest implementation of the project. It's single core.
 * The only small optimization is downsampling and upsampling at the same time as convolving, but the original clean functions are included in this same file
 * 
 * Local laplacian filter works as follows:
 * - Create a gaussian pyramid starting from the source image
 * - For each pixel, for each layer of the gauss pyramid:
 * -- take the current pixel G0 from the original gaussian pyramid
 * -- cut a subregion from the source image with a dimension proportional to the layer's dimension near the pixel
 * -- apply a remap function to this subregion using G0
 * -- create a gaussian pyramid over this subregion
 * -- create a laplacian pyramid over the previous gaussian pyramid
 * -- copy the pixel placed at the correct coordinates respect to the original pixel, on the current layer of the laplacian pyramid, to the current layer of the output laplacian pyramid
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
void llf(Image3 *img, float sigma, float alpha, float beta, uint8_t nLevels, WorkingBuffers *workingBuffers){
	uint32_t width = img->width, height = img->height;
	nLevels = llf_min(nLevels, 5); //Clamps the number of levels
	nLevels = llf_max(nLevels, 3);//int(ceil(std::abs(std::log2(llf_min(width, height)) - 3))) + 2;
	Kernel filter = workingBuffers->filter;
	Pyramid gaussPyramid = workingBuffers->gaussPyramid;
	Pyramid outputLaplacian = workingBuffers->outputLaplacian;
	Pyramid bufferGaussPyramid = workingBuffers->bufferGaussPyramid;
	Pyramid bufferLaplacianPyramid = workingBuffers->bufferLaplacianPyramid; //Get the working buffers

	TimeData timeData; //Variables used to cout the elapsed computation time. Compile with the preprocessor flag SHOW_TIME_STATS = 1 to show these stats
	TimeCounter passed = 0;

	//print("Creating first gauss pyramid");
	startTimerCounter(timeData);
	gaussianPyramid(gaussPyramid, img, nLevels, filter); //Creates a gaussian pyramid starting from the source image
	stopTimerCounter(timeData, passed);
	//print("Entering main loop");
	startTimerCounter(timeData);
	for(uint8_t lev = 0; lev < nLevels; lev++){ //For each layer of the gaussian pyramid
		//printff("laplacian inner loop %d/%d\n", lev, (nLevels - 1));
		Image3 *currentGaussLevel = gaussPyramid[lev];
		uint32_t gaussianWidth = currentGaussLevel->width, gaussianHeight = currentGaussLevel->height;
		uint32_t subregionDimension = 3 * ((1 << (lev + 2)) - 1) / 2; //Calc the subregion dimension

		for(uint32_t y = 0; y < gaussianHeight; y++){

			//no fuckin clues what this calcs are
			int32_t full_res_y = (1 << lev) * y;
			int32_t roi_y0 = full_res_y - subregionDimension;
			int32_t roi_y1 = full_res_y + subregionDimension + 1;
			int32_t base_y = llf_max(0, roi_y0);
			int32_t end_y = llf_min(roi_y1, height);
			int32_t full_res_roi_y = full_res_y - base_y;
			int32_t full_res_roi_yShifted = full_res_roi_y >> lev;

			for(uint32_t x = 0; x < gaussianWidth; x++){ //For each pixel
				//no fuckin clues what this calcs are PT2
				int32_t full_res_x = (1 << lev) * x;
				int32_t roi_x0 = full_res_x - subregionDimension;
				int32_t roi_x1 = full_res_x + subregionDimension + 1;
				int32_t base_x = llf_max(0, roi_x0);
				int32_t end_x = llf_min(roi_x1, width);
				int32_t full_res_roi_x = full_res_x - base_x;

				Pixel3 g0 = *getPixel3(currentGaussLevel, x, y); //Get the pixel from the current gaussian level
				subimage3(bufferLaplacianPyramid[0], img, base_x, end_x, base_y, end_y); //Cut a subregion from the source image using bufferLaplacianPyramid[0] as temp buffer
				remap(bufferLaplacianPyramid[0], g0, sigma, alpha, beta); //Remap the subregion using g0
				uint8_t currentNLevels = lev + 1; //Pyramids has lev+1 layers!
				gaussianPyramid(bufferGaussPyramid, bufferLaplacianPyramid[0], currentNLevels, filter); //Build a gauss pyramid over the subregion
				laplacianPyramid(bufferLaplacianPyramid, bufferGaussPyramid, currentNLevels, filter); //Build a laplacian pyramid over the gauss pyramid

				setPixel3(outputLaplacian[lev], x, y, getPixel3(bufferLaplacianPyramid[lev], full_res_roi_x >> lev, full_res_roi_yShifted)); //copy the correct pixel from the local laplacian pyramid to the output laplacian pyramid
			}
		}
	}
	imgcpy3(outputLaplacian[nLevels], gaussPyramid[nLevels]); //Copy the smallest layer of the gauss pyramid over the output laplacian pyramid
	//print("Collapsing");
	collapse(img, outputLaplacian, nLevels, filter); //Collapse the output laplacian pyramid over the dest image
	stopTimerCounter(timeData, passed);
	#ifdef SHOW_TIME_STATS
		#if ON_WINDOWS
			printff("Total time: %dms\n", passed);
		#else
			printff("Total time: %lums\n", passed);
		#endif
	#endif

	clampImage3(img); //Clamp the dest image to put all the pixel in the [0;1] bounds
}

/**
 * @brief allocates the data structures needed for llf's processing
 * 
 * @param workingBuffers non-allocated data structures
 * @param width width of the pyramids
 * @param height height of the pyramids
 * @param nLevels number of layers of the pyramids
 */
void initWorkingBuffers(WorkingBuffers *workingBuffers, uint32_t width, uint32_t height, uint8_t nLevels){
	printff("Creating pyramids %dx%d @ %d levels\n", width, height, nLevels);
	workingBuffers->bufferLaplacianPyramid = createPyramid(width, height, nLevels);
	workingBuffers->bufferGaussPyramid = createPyramid(width, height, nLevels);
	workingBuffers->outputLaplacian = createPyramid(width, height, nLevels);
	workingBuffers->gaussPyramid = createPyramid(width, height, nLevels);
	workingBuffers->filter = createFilter();
}
/**
 * @brief destroy the data structures needed for llf's processing
 * 
 * @param workingBuffers allocated data structures
 * @param nLevels number of layers of the already allocated pyramids
 */
void destroyWorkingBuffers(WorkingBuffers *workingBuffers, uint8_t nLevels){
	destroyPyramid(&(workingBuffers->bufferLaplacianPyramid), nLevels);
	destroyPyramid(&(workingBuffers->bufferGaussPyramid), nLevels);
	destroyPyramid(&(workingBuffers->outputLaplacian), nLevels);
	destroyPyramid(&(workingBuffers->gaussPyramid), nLevels);
	destroyFilter(&(workingBuffers->filter));
}

