#include "openmp.h"

void upsampleConvolve(Image3 *dest, Image3 *source, Kernel kernel){
	const uint32_t smallWidth = source->width, smallHeight = source->height;
	const uint32_t uppedW = smallWidth << 1;
	const uint32_t uppedH = smallHeight << 1;
	dest->width = uppedW;
	dest->height = uppedH;
	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;

	for (uint32_t j = 0; j < uppedH; j++) {
		for (uint32_t i = 0; i < uppedW; i++) {
			Pixel3 c = zero3vect;
			for (uint32_t y = 0; y < rows; y++) {
                int32_t jy = (j + ystart + y) / 2;
				for (uint32_t x = 0; x < cols; x++) {
                    int32_t ix = (i + xstart + x) / 2;
                    if (ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight) {
						float kern_elem = kernel[getKernelPosition(x, y)];
						Pixel3 px = *getPixel3(source, ix, jy);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					} else {
						float kern_elem = kernel[getKernelPosition(x, y)];
						Pixel3 px = *getPixel3(source, i / 2, j / 2);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					}
				}
			}
			setPixel3(dest, i, j, &c);
		}
	}
}
Pixel3 upsampleConvolveSubtractSinglePixel(Image3 *source, Pixel3 *gaussPx, Kernel kernel, uint32_t i, uint32_t j){
	//This function will only work on the interested subregion get the pixel we're gonna place in the final laplacian pyramid
	uint32_t smallWidth = source->width, smallHeight = source->height;
	Pixel3* srcPx = source->pixels;
	const int32_t  xstart = -1 * KERNEL_DIMENSION / 2;
	const int32_t  ystart = -1 * KERNEL_DIMENSION / 2;
	
	Pixel3 ups = zero3vect;
	for (uint32_t y = 0; y < KERNEL_DIMENSION; y++) {
		int32_t jy = (j + ystart + y) / 2;
		for (uint32_t x = 0; x < KERNEL_DIMENSION; x++) {
			int32_t ix = (i + xstart + x) / 2;
			if (ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight) {
				float kern_elem = kernel[getKernelPosition(x, y)];
				Pixel3 px = *getPixel3(source, ix, jy);

				ups.x += px.x * kern_elem;
				ups.y += px.y * kern_elem;
				ups.z += px.z * kern_elem;
			} else {
				float kern_elem = kernel[getKernelPosition(x, y)];
				Pixel3 px = *getPixel3(source, i / 2, j / 2);

				ups.x += px.x * kern_elem;
				ups.y += px.y * kern_elem;
				ups.z += px.z * kern_elem;
			}
		}
	}
	vec3Sub(ups, *gaussPx, ups);
	return ups;
}
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

	#pragma omp parallel for num_threads(nThreads) schedule(static)
	for (int32_t idx = 0; idx < dim; idx++) {
		uint32_t i = idx % uppedW, j = idx / uppedW;

		Pixel3 c = zero3vect;
		for (uint32_t y = 0; y < rows; y++) {
			int32_t jy = (j + ystart + y) / 2;
			for (uint32_t x = 0; x < cols; x++) {
				int32_t ix = (i + xstart + x) / 2;
				if (ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight) {
					float kern_elem = kernel[getKernelPosition(x, y)];
					Pixel3 px = *getPixel3(source, ix, jy);

					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
				} else {
					float kern_elem = kernel[getKernelPosition(x, y)];
					Pixel3 px = *getPixel3(source, i / 2, j / 2);

					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
				}
			}
		}
		setPixel3(dest, i, j, &c);
	}
}

void laplacianPyramid(Pyramid laplacian, Pyramid tempGauss, uint8_t nLevels, Kernel filter){
	for(uint8_t i = 0; i < nLevels; i++){
		Image3 *upsampled = laplacian[i];
		upsampleConvolve(upsampled, tempGauss[i + 1], filter);

		Image3 *current = tempGauss[i];
		uint32_t yEnd = llf_min(current->height, upsampled->height);
		uint32_t xEnd = llf_min(current->width, upsampled->width);
		for (uint32_t y = 0; y < yEnd; y++){
			for (uint32_t x = 0; x < xEnd; x++){
				Pixel3 *upsPtr = getPixel3(upsampled, x, y);
				Pixel3 ups = *upsPtr;
				Pixel3 crr = *getPixel3(current, x, y);

				vec3Sub(*upsPtr, crr, ups);
			}
		}
	}
	imgcpy3(laplacian[nLevels], tempGauss[nLevels]);
}

void collapse(Image3 *dest, Pyramid laplacianPyr, uint8_t nLevels, Kernel filter, const uint8_t nThreads){
	Pixel3 *destPxs = dest->pixels;
	for(int8_t lev = nLevels; lev > 1; lev--){ //Using dest as a temp buffer
		Image3 *currentLevel = laplacianPyr[lev], *biggerLevel = laplacianPyr[lev - 1];
		Pixel3 *biggerLevelPxs = biggerLevel->pixels;

		upsampleConvolve_parallel(dest, currentLevel, filter, nThreads);
		uint32_t sizeUpsampled = llf_min(dest->width, biggerLevel->width) * llf_min(dest->height, biggerLevel->height);
		#pragma omp parallel for num_threads(nThreads) schedule(static, 8)
		for(int32_t px = 0; px < sizeUpsampled; px++)	
			vec3Add(biggerLevelPxs[px], destPxs[px], biggerLevelPxs[px]);
		biggerLevel->width = dest->width;
		biggerLevel->height = dest->height; //This could cause disalignment problem
	}
	Image3 *currentLevel = laplacianPyr[1], *biggerLevel = laplacianPyr[0];
	Pixel3 *biggerLevelPxs = biggerLevel->pixels;

	upsampleConvolve_parallel(dest, currentLevel, filter, nThreads);
	uint32_t sizeUpsampled = llf_min(dest->width, biggerLevel->width) * llf_min(dest->height, biggerLevel->height);
	#pragma omp parallel for num_threads(nThreads) schedule(static, 8)
	for(int32_t px = 0; px < sizeUpsampled; px++)
		vec3Add(destPxs[px], destPxs[px], biggerLevelPxs[px]);
}

void downsampleConvolve(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter){
	const uint32_t originalW = *width, originalH = *height;
	*width /= 2;
	*height /= 2;
	dest->width = *width;
	dest->height = *height;
	const int32_t startingX = originalW & 1;
	const int32_t startingY = originalH & 1;
	const int8_t  rows = KERNEL_DIMENSION;
	const int8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;

	for (uint32_t j = startingY; j < originalH; j += 2) {
		for (uint32_t i = startingX; i < originalW; i += 2) {
			Pixel3 c = zero3vect;
			for (uint32_t y = 0; y < rows; y++) {
				int32_t jy = j + (ystart + y) * 2 - startingY;
				for (uint32_t x = 0; x < cols; x++) {
					int32_t ix = i + (xstart + x) * 2 - startingX;

					if (ix >= 0 && ix < originalW && jy >= 0 && jy < originalH) {
						float kern_elem = filter[getKernelPosition(x, y)];
						Pixel3 px = *getPixel3(source, ix, jy);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					} else {
						
						float kern_elem = filter[getKernelPosition(x, y)];
						Pixel3 px = *getPixel3(source, i - startingX, j - startingY);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					}
				}
			}
			setPixel3(dest, i / 2, j / 2, &c);
		}
	}
}
void downsampleConvolve_parallel(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter, const uint8_t nThreads){
	uint32_t originalW = *width, originalH = *height;
	*width /= 2;
	*height /= 2;
	dest->width = *width;
	dest->height = *height;
	const int32_t startingX = originalW & 1;
	const int32_t startingY = originalH & 1;
	const int8_t  rows = KERNEL_DIMENSION;
	const int8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;
	originalW -= startingX;
	const uint32_t dim = (originalH - startingY * 2) * originalW; //not *2 on w because we need one extra pixel at the end of the line for the +=2 to work

	#pragma omp parallel for num_threads(nThreads) schedule(static)
	for(int32_t idx = 0; idx < dim; idx += 2){
		uint32_t i = (idx % originalW) + startingX, j = (idx / originalW) + startingY;

		Pixel3 c = zero3vect;
		for (uint32_t y = 0; y < rows; y++) {
			int32_t jy = j + (ystart + y) * 2 - startingY;
			for (uint32_t x = 0; x < cols; x++) {
				int32_t ix = i + (xstart + x) * 2 - startingX;

				if (ix >= 0 && ix < originalW && jy >= 0 && jy < originalH) {
					float kern_elem = filter[getKernelPosition(x, y)];
					Pixel3 px = *getPixel3(source, ix, jy);

					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
				} else {
					
					float kern_elem = filter[getKernelPosition(x, y)];
					Pixel3 px = *getPixel3(source, i - startingX, j - startingY);

					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
				}
			}
		}
		setPixel3(dest, i / 2, j / 2, &c);
	}
}

void gaussianPyramid(Pyramid outPyr, Image3 *inImg, uint8_t nLevels, Kernel filter){
	imgcpy3(outPyr[0], inImg);
	uint32_t width = inImg->width, height = inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve(outPyr[1], inImg, &width, &height, filter);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve(outPyr[i + 1], outPyr[i], &width, &height, filter);
}
void gaussianPyramid_fast(Pyramid outPyr, uint8_t nLevels, Kernel filter){
	//This function assumes that in outPyr[0] there's already the original image copied
	uint32_t width = outPyr[0]->width, height = outPyr[0]->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve(outPyr[1], outPyr[0], &width, &height, filter);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve(outPyr[i + 1], outPyr[i], &width, &height, filter);
}
void gaussianPyramid_parallel(Pyramid outPyr, Image3 *inImg, uint8_t nLevels, Kernel filter, const uint8_t nThreads){
	imgcpy3_parallel(outPyr[0], inImg, nThreads);
	uint32_t width = inImg->width, height = inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve_parallel(outPyr[1], inImg, &width, &height, filter, nThreads);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve_parallel(outPyr[i + 1], outPyr[i], &width, &height, filter, nThreads);
}


void llf(Image3 *img, float sigma, float alpha, float beta, uint8_t nLevels, const uint8_t nThreads, WorkingBuffers *workingBuffers){
	uint32_t width = img->width, height = img->height;
	nLevels = llf_min(nLevels, 5);
	nLevels = llf_max(nLevels, 3);//int(ceil(std::abs(std::log2(llf_min(width, height)) - 3))) + 2;
	Kernel filter = workingBuffers->filter;

	Pyramid gaussPyramid = workingBuffers->gaussPyramid;
	Pyramid outputLaplacian = workingBuffers->outputLaplacian;

	TimeData timeData;
	TimeCounter passed = 0;

	//print("Creating first gauss pyramid");
	startTimerCounter(timeData);
	gaussianPyramid_parallel(gaussPyramid, img, nLevels, filter, nThreads);
	stopTimerCounter(timeData, passed);
	//print("Entering main loop");
	// Sadly, due to approxxximation in the downsample function, I can't use sum to calculate the pyramid dimension :(
	//uint32_t t = (0b100 << (nLevels * 2));
	//uint32_t end = (img->width * img->height * ((t - 1) / 3)) / (t / 4); //sum[i=0, n] D / 4^i
	uint32_t *pyrDimensions = workingBuffers->pyrDimensions;

	Pyramid *bArr = workingBuffers->bArr;
	CurrentLevelInfo *cliArr = workingBuffers->cliArr;
	#pragma omp parallel num_threads(nThreads)
	{
		initLevelInfo(&(cliArr[getThreadId()]), pyrDimensions, gaussPyramid);
	}

	startTimerCounter(timeData);
	#pragma omp parallel for num_threads(nThreads) schedule(dynamic)
	for(int32_t idx = 0; idx < workingBuffers->end; idx++){
		int threadId = getThreadId();
		CurrentLevelInfo *cli = &(cliArr[threadId]);
		Pyramid bufferGaussPyramid = workingBuffers->bArr[threadId];

		if(idx >= cli->nextLevelDimension) //Assuming ofc that idk only goes up for each thread
			updateLevelInfo(cli, pyrDimensions, gaussPyramid);
		int32_t localIdx = idx - cli->prevLevelDimension;

		uint8_t lev = cli->lev;
		Image3 *currentGaussLevel = cli->currentGaussLevel;
		uint32_t gaussianWidth = cli->width;
		uint32_t subregionDimension = cli->subregionDimension;
		uint32_t x = localIdx % gaussianWidth, y = localIdx / gaussianWidth;
		
		//no fuckin clues what this calcs are
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

		Pixel3 g0 = *getPixel3(currentGaussLevel, x, y);
		subimage3(bufferGaussPyramid[0], img, base_x, end_x, cli->base_y, cli->end_y); //Using b.bufferLaplacianPyramid[0] as temp buffer
		remap(bufferGaussPyramid[0], g0, sigma, alpha, beta);
		uint8_t currentNLevels = cli->currentNLevels;
		gaussianPyramid_fast(bufferGaussPyramid, currentNLevels, filter);
		Pixel3 *gausPx = getPixel3(bufferGaussPyramid[currentNLevels - 1], full_res_roi_xShifted, cli->full_res_roi_yShifted);
		Pixel3 outPx = upsampleConvolveSubtractSinglePixel(bufferGaussPyramid[currentNLevels], gausPx, filter, full_res_roi_xShifted, cli->full_res_roi_yShifted);

		setPixel3(outputLaplacian[lev], x, y, &outPx); //idk why i had to shift those
	}

	imgcpy3_parallel(outputLaplacian[nLevels], gaussPyramid[nLevels], nThreads);
	//print("Collapsing");
	collapse(img, outputLaplacian, nLevels, filter, nThreads);
	stopTimerCounter(timeData, passed);
	#ifdef SHOW_TIME_STATS
		#if ON_WINDOWS
			printff("Total time: %dms\n", passed);
		#else
			printff("Total time: %lums\n", passed);
		#endif
	#endif

	clampImage3(img);
}

void initWorkingBuffers(WorkingBuffers *workingBuffers, uint32_t width, uint32_t height, uint8_t nLevels, uint8_t nThreads){
	workingBuffers->outputLaplacian = createPyramid(width, height, nLevels);
	workingBuffers->gaussPyramid = createPyramid(width, height, nLevels);
	workingBuffers->pyrDimensions = (uint32_t *) malloc((nLevels + 1) * sizeof(uint32_t));
	workingBuffers->end = 0;
	for(uint8_t i = 0; i < nLevels; i++){
		Image3 *lev = workingBuffers->gaussPyramid[i];
		uint32_t dim = lev->width * lev->height;
		workingBuffers->pyrDimensions[i] = dim;
		workingBuffers->end += dim;
	}
	workingBuffers->pyrDimensions[nLevels] = workingBuffers->gaussPyramid[nLevels]->width * workingBuffers->gaussPyramid[nLevels]->height;
	workingBuffers->cliArr = (CurrentLevelInfo *) malloc(nThreads * sizeof(CurrentLevelInfo));
	workingBuffers->bArr = (Pyramid *) malloc(nThreads * sizeof(Pyramid));
	for(uint32_t i = 0; i < nThreads; i++)
		workingBuffers->bArr[i] = createPyramid(width, height, nLevels);
	workingBuffers->filter = createFilter();
}
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