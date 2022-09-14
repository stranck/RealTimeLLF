#include "../utils/imageutils.h"
#include "../utils/extramath.h"
#include "../utils/llfUtils.h"
#include "../utils/structs.h"
#include "../utils/vects.h"
#include "../utils/utils.h"
#include "openmpStructs.h"
#include "openmpUtils.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "../utils/test/testimage.h"

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

void gaussianPyramid(Pyramid outPyr, Image3 *inImg, uint8_t nLevels, Kernel filter){
	imgcpy3(outPyr[0], inImg);
	uint32_t width = inImg->width, height = inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve(outPyr[1], inImg, &width, &height, filter);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve(outPyr[i + 1], outPyr[i], &width, &height, filter);
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
	for(uint32_t idx = 0; idx < dim; idx += 2){
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
	for (uint32_t idx = 0; idx < dim; idx++) {
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

void gaussianPyramid_parallel(Pyramid outPyr, Image3 *inImg, uint8_t nLevels, Kernel filter, const uint8_t nThreads){
	imgcpy3_parallel(outPyr[0], inImg, nThreads);
	uint32_t width = inImg->width, height = inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve_parallel(outPyr[1], inImg, &width, &height, filter, nThreads);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve_parallel(outPyr[i + 1], outPyr[i], &width, &height, filter, nThreads);
}

void collapse(Image3 *dest, Pyramid laplacianPyr, uint8_t nLevels, Kernel filter, const uint8_t nThreads){
	Pixel3 *destPxs = dest->pixels;
	for(int8_t lev = nLevels; lev > 1; lev--){ //Using dest as a temp buffer
		Image3 *currentLevel = laplacianPyr[lev], *biggerLevel = laplacianPyr[lev - 1];
		Pixel3 *biggerLevelPxs = biggerLevel->pixels;

		upsampleConvolve_parallel(dest, currentLevel, filter, nThreads);
		uint32_t sizeUpsampled = llf_min(dest->width, biggerLevel->width) * llf_min(dest->height, biggerLevel->height);
		#pragma omp parallel for num_threads(nThreads) schedule(static, 8)
		for(uint32_t px = 0; px < sizeUpsampled; px++)	
			vec3Add(biggerLevelPxs[px], destPxs[px], biggerLevelPxs[px]);
		biggerLevel->width = dest->width;
		biggerLevel->height = dest->height; //This could cause disalignment problem
	}
	Image3 *currentLevel = laplacianPyr[1], *biggerLevel = laplacianPyr[0];
	Pixel3 *biggerLevelPxs = biggerLevel->pixels;

	upsampleConvolve_parallel(dest, currentLevel, filter, nThreads);
	uint32_t sizeUpsampled = llf_min(dest->width, biggerLevel->width) * llf_min(dest->height, biggerLevel->height);
	#pragma omp parallel for num_threads(nThreads) schedule(static, 8)
	for(uint32_t px = 0; px < sizeUpsampled; px++)
		vec3Add(destPxs[px], destPxs[px], biggerLevelPxs[px]);
}

void llf(Image3 *img, float sigma, float alpha, float beta, uint8_t nLevels, const uint8_t nThreads){
	uint32_t width = img->width, height = img->height;
	nLevels = llf_min(nLevels, 5);
	nLevels = llf_max(nLevels, 3);//int(ceil(std::abs(std::log2(llf_min(width, height)) - 3))) + 2;
	Kernel filter = createFilter();

	Pyramid gaussPyramid = createPyramid(width, height, nLevels);
	Pyramid outputLaplacian = createPyramid(width, height, nLevels);

	TimeData timeData;
	TimeCounter passed = 0;

	print("Creating first gauss pyramid");
	startTimerCounter(timeData);
	gaussianPyramid_parallel(gaussPyramid, img, nLevels, filter, nThreads);
	stopTimerCounter(timeData, passed);
	print("Entering main loop");
	// Sadly, due to approxxximation in the downsample function, I can't use sum to calculate the pyramid dimension :(
	//uint32_t t = (0b100 << (nLevels * 2));
	//uint32_t end = (img->width * img->height * ((t - 1) / 3)) / (t / 4); //sum[i=0, n] D / 4^i
	uint32_t end = 0;
	uint32_t *pyrDimensions = (uint32_t *) allocStack((nLevels + 1) * sizeof(uint32_t));
	for(uint8_t i = 0; i < nLevels; i++){
		Image3 *lev = gaussPyramid[i];
		uint32_t dim = lev->width * lev->height;
		pyrDimensions[i] = dim;
		end += dim;
	}
	pyrDimensions[nLevels] = gaussPyramid[nLevels]->width * gaussPyramid[nLevels]->height;

	Buffers *bArr = (Buffers *) allocStack(nThreads * sizeof(Buffers));
	CurrentLevelInfo *cliArr = (CurrentLevelInfo *) allocStack(nThreads * sizeof(CurrentLevelInfo));
	#pragma omp parallel num_threads(nThreads)
	{
		int threadId = getThreadId();
		createBuffers(bArr[threadId], width, height, nLevels);
		initLevelInfo(&(cliArr[threadId]), pyrDimensions, gaussPyramid);
	}

	startTimerCounter(timeData);
	#pragma omp parallel for num_threads(nThreads) schedule(dynamic)
	for(uint32_t idx = 0; idx < end; idx++){
		int threadId = getThreadId();
		CurrentLevelInfo *cli = &(cliArr[threadId]);
		Buffers *b = &(bArr[threadId]);

		if(idx >= cli->nextLevelDimension) //Assuming ofc that idk only goes up for each thread
			updateLevelInfo(cli, pyrDimensions, gaussPyramid);
		uint32_t localIdx = idx - cli->prevLevelDimension;

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

		Pixel3 g0 = *getPixel3(currentGaussLevel, x, y);
		subimage3(b->bufferLaplacianPyramid[0], img, base_x, end_x, cli->base_y, cli->end_y); //Using b.bufferLaplacianPyramid[0] as temp buffer
		remap(b->bufferLaplacianPyramid[0], g0, sigma, alpha, beta);
		uint8_t currentNLevels = cli->currentNLevels;
		gaussianPyramid(b->bufferGaussPyramid, b->bufferLaplacianPyramid[0], currentNLevels, filter);
		laplacianPyramid(b->bufferLaplacianPyramid, b->bufferGaussPyramid, currentNLevels, filter);

		setPixel3(outputLaplacian[lev], x, y, getPixel3(b->bufferLaplacianPyramid[lev], full_res_roi_x >> lev, cli->full_res_roi_yShifted)); //idk why i had to shift those
	}

	imgcpy3_parallel(outputLaplacian[nLevels], gaussPyramid[nLevels], nThreads);
	print("Collapsing");
	collapse(img, outputLaplacian, nLevels, filter, nThreads);
	stopTimerCounter(timeData, passed);
	#ifdef SHOW_TIME_STATS
		printff("Total time: %lums\n", passed);
	#endif
	destroyPyramid(&gaussPyramid, nLevels);
	destroyPyramid(&outputLaplacian, nLevels);
	for(uint8_t i = 0; i < nThreads; i++){
		destroyPyramid(&(bArr[i].bufferGaussPyramid), nLevels);
		destroyPyramid(&(bArr[i].bufferLaplacianPyramid), nLevels);
	}
	destroyFilter(&filter);
}

/*
.\bin\ycolorgrade --image tests/flower.png --llf --levels 3 --sigma 0.35 --alpha 0.4 --beta 5
Yocto single core: 21213ms 
Yocto multicore: 1906ms
This single core: 11707ms
This multi core: 836ms
*/
int main(){
	Image4 *img4 = getStaticImage4();
	Image3 *img = image4to3(img4);
	AlphaMap map = getAlphaMap(img4);
	destroyImage4(&img4);

	llf(img, 0.35, 0.4, 5, 3, 24);

	clampImage3(img);
	img4 = image3to4AlphaMap(img, map);
	destroyImage3(&img);
	printStaticImage4(img4);
	destroyImage4(&img4);
}