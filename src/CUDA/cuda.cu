//#define CUDA_INCLUDE //had to do this to fix vs code intellisense doing random stuff

#define MAX_LAYERS 5
#define MAX_PYR_LAYER 3 * ((1 << (MAX_LAYERS + 2)) - 1)

#include "../utils/imageutils.h"
#include "../utils/extramath.h"
#include "../utils/llfUtils.h"
#include "../utils/structs.h"
#include "../utils/vects.h"
#include "../utils/utils.h"
#include "cudaStructs.cuh"
#include "cudaUtils.cuh"
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#ifdef CUDA_INCLUDE
	#include <cuda.h>
	#include <cuda_runtime.h>
#endif

#include "../utils/test/testimage.h"

/*void testShit(){
	const uint32_t originalH = 31;
	const int32_t startingY = originalH & 1;
	const int8_t  rows = KERNEL_DIMENSION;
	const int32_t  ystart = -1 * rows / 2;

	for (uint32_t j = startingY; j < originalH; j += 2) {
		for (uint32_t y = 0; y < rows; y++) {
			int32_t jy = j + (ystart + y) * 2 - startingY;
			printff("J: %d\t\t Y: %d\t\t JY: %d\n", j, y, jy);
		}
	}
}*/

__device__ void upsampleConvolve(Image3 *dest, Image3 *source, Kernel kernel){
	__shared__ Pixel3 ds_upsampled[MAX_PYR_LAYER * MAX_PYR_LAYER];
	
	uint32_t smallWidth = source->width, smallHeight = source->height;
	uint32_t uppedW = smallWidth << 1;
	uint32_t uppedH = smallHeight << 1;
	if(threadIdx.x == 0){
		dest->width = uppedW;
		dest->height = uppedH;
	}
	__syncthreads();
	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;
	
	uint32_t dim = smallWidth * smallHeight;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t x = idx % smallWidth, y = idx / smallWidth;
			ds_upsampled[y * smallWidth + x] = *getPixel3(source, x, y);
		}
	}
	__syncthreads();

	dim = uppedW * uppedH;
	max = dim / blockDim.x;

	//for (uint32_t j = 0; j < uppedH; j++) {
	//	for (uint32_t i = 0; i < uppedW; i++) {
	for(uint32_t li = 0; li <= max; li++){
		uint32_t idx = li * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t i = idx % uppedW, j = idx / uppedW;

			Pixel3 c = zero3f;
			for (uint32_t y = 0; y < rows; y++) {
                int32_t jy = (j + ystart + y) / 2;
				for (uint32_t x = 0; x < cols; x++) {
                    int32_t ix = (i + xstart + x) / 2;

					int32_t oob = ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight;
					int32_t fi = ix * oob + (i / 2) * (1 - oob), fj = jy * oob + (j / 2) * (1 - oob);

					double kern_elem = kernel[getKernelPosition(x, y)];
					Pixel3 px = ds_upsampled[fj * uppedW + fi]; //*getPixel3(source, ix, jy);
					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
                    /*if (ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight) {
						double kern_elem = kernel[getKernelPosition(x, y)];
						Pixel3 px = *getPixel3(source, ix, jy);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					} else {
						double kern_elem = kernel[getKernelPosition(x, y)];
						Pixel3 px = *getPixel3(source, i / 2, j / 2);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					}*/
				}
			}
			setPixel3(dest, i, j, &c);
		}
	}
	__syncthreads();
}

__device__ void downsampleConvolve(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter){
	__shared__ Pixel3 ds_downsampled[MAX_PYR_LAYER * MAX_PYR_LAYER];
	const uint32_t originalW = *width, originalH = *height;
	uint32_t lcl_width = *width / 2;
	if(threadIdx.x == 0){
		*width = lcl_width;
		*height /= 2;
		dest->width = lcl_width;
		dest->height = *height;
	}
	__syncthreads();
	uint32_t startingX = originalW & 1;
	uint32_t startingY = originalH & 1;
	
	uint32_t dim = lcl_width * *height;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;

		if(idx < dim){
			uint32_t x = idx % lcl_width, y = idx / lcl_width;
			ds_downsampled[y * lcl_width + x] = *getPixel3(source, (x * 2) - startingX, (y * 2) - startingY);
		}
	}
	__syncthreads();
	/*for(uint32_t y = startingY; y < originalH; y += 2) {
		for(uint32_t x = startingX; x < originalW; x += 2) {
			ds_downsampled[(y / 2) * (*width) + (x / 2)] = *getPixel3(source, x - startingX, y - startingY);
		}
	}*/

	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;

	//for (int32_t j = 0; j < dest->height; j++) {
	//	for (int32_t i = 0; i < dest->width; i++) {
	for(uint32_t li = 0; li <= max; li++){
		uint32_t idx = li * blockDim.x + threadIdx.x;

		if(idx < dim){
			uint32_t i = idx % lcl_width, j = idx / lcl_width;
			Pixel3 c = zero3f;
			for (int32_t y = 0; y < rows; y++) {
				int32_t jy = j + ystart + y;
				for (int32_t x = 0; x < cols; x++) {
					int32_t ix = i + xstart + x;

					int32_t oob = ix >= 0 && ix < dest->width && jy >= 0 && jy < dest->height;
					int32_t fi = ix * oob + i * (1 - oob), fj = jy * oob + j * (1 - oob);

					double kern_elem = filter[getKernelPosition(x, y)];
					Pixel3 px = ds_downsampled[fj * lcl_width + fi]; //*getPixel3(source, fx, fj);
					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
				}
			}
			setPixel3(dest, i, j, &c);
		}
	}
	__syncthreads();
}

__global__ void gaussianPyramid(Pyramid d_outPyr, Image3 *d_inImg, uint8_t nLevels, Kernel d_filter){
	d_imgcpy3(d_outPyr[0], d_inImg);
	uint32_t width = d_inImg->width, height = d_inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve(d_outPyr[1], d_inImg, &width, &height, d_filter);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve(d_outPyr[i + 1], d_outPyr[i], &width, &height, d_filter);
}

/*__host__ void llf(Image3 *img, double sigma, double alpha, double beta, uint8_t nLevels, const uint8_t nThreads){
	uint32_t width = img->width, height = img->height;
	nLevels = min(nLevels, 5);
	nLevels = max(nLevels, 3);//int(ceil(std::abs(std::log2(min(width, height)) - 3))) + 2;
	Kernel filter = createFilterDevice();

	Pyramid d_gaussPyramid = createPyramidDevice(width, height, nLevels);
	Pyramid d_outputLaplacian = createPyramidDevice(width, height, nLevels);

	print("Creating first gauss pyramid");
	gaussianPyramid(d_gaussPyramid, img, nLevels, filter);
	print("Entering main loop");
	// Sadly, due to approxxximation in the downsample function, I can't use sum to calculate the pyramid dimension :(
	//uint32_t t = (0b100 << (nLevels * 2));
	//uint32_t end = (img->width * img->height * ((t - 1) / 3)) / (t / 4); //sum[i=0, n] D / 4^i
	uint32_t end = 0;
	uint32_t pyrDimensions[nLevels + 1];
	for(uint8_t i = 0; i < nLevels; i++){
		Image3 *lev = gaussPyramid[i];
		uint32_t dim = lev->width * lev->height;
		pyrDimensions[i] = dim;
		end += dim;
	}
	pyrDimensions[nLevels] = gaussPyramid[nLevels]->width * gaussPyramid[nLevels]->height;

	Buffers bArr[nThreads];
	CurrentLevelInfo cliArr[nThreads];
	#pragma omp parallel num_threads(nThreads)
	{
		int threadId = getThreadId();
		bArr[threadId] = createBuffers(width, height, nLevels);
		initLevelInfo(&(cliArr[threadId]), pyrDimensions, gaussPyramid);
	}

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
			cli->end_y = min(roi_y1, height);
			uint32_t full_res_roi_y = full_res_y - cli->base_y;
			cli->full_res_roi_yShifted = full_res_roi_y >> lev;
			cli->oldY = y;
		}

		uint32_t full_res_x = (1 << lev) * x;
		uint32_t roi_x1 = full_res_x + subregionDimension + 1;
		uint32_t base_x = subregionDimension > full_res_x ? 0 : full_res_x - subregionDimension;
		uint32_t end_x = min(roi_x1, width);
		uint32_t full_res_roi_x = full_res_x - base_x;

		Pixel3 g0 = *getPixel3(currentGaussLevel, x, y);
		subimage3(b->bufferLaplacianPyramid[0], img, base_x, end_x, cli->base_y, cli->end_y); //Using b.bufferLaplacianPyramid[0] as temp buffer
		remap(b->bufferLaplacianPyramid[0], g0, sigma, alpha, beta);
		uint8_t currentNLevels = cli->currentNLevels;
		gaussianPyramid(b->bufferGaussPyramid, b->bufferLaplacianPyramid[0], currentNLevels, filter);
		laplacianPyramid(b->bufferLaplacianPyramid, b->bufferGaussPyramid, currentNLevels, filter);

		setPixel3(outputLaplacian[lev], x, y, getPixel3(b->bufferLaplacianPyramid[lev], full_res_roi_x >> lev, cli->full_res_roi_yShifted)); //idk why i had to shift those
	}

	imgcpy3(outputLaplacian[nLevels], gaussPyramid[nLevels]);
	print("Collapsing");
	collapse(img, outputLaplacian, nLevels, filter);

	destroyPyramid(&gaussPyramid, nLevels);
	destroyPyramid(&outputLaplacian, nLevels);
	for(uint8_t i = 0; i < nThreads; i++){
		destroyPyramid(&(bArr[i].bufferGaussPyramid), nLevels);
		destroyPyramid(&(bArr[i].bufferLaplacianPyramid), nLevels);
	}
	destroyFilter(&filter);
}*/

uint32_t getPixelNoPerPyramid(uint8_t nLevels){
	uint32_t subregionDimension = 3 * ((1 << (nLevels + 2)) - 1);
	uint32_t totalPixels = 0;
	for(uint8_t i = 0; i <= nLevels; i++){
		totalPixels += (subregionDimension * subregionDimension);
		subregionDimension = subregionDimension / 2 + (subregionDimension & 1);
	}
	return totalPixels;
}

int main(){
	Image4 *img4 = getStaticImage4();
	Image3 *img = image4to3(img4);
	AlphaMap map = getAlphaMap(img4);
	destroyImage4(&img4);

	//llf(img, 0.35, 0.4, 5, 3, 24);

	clampImage3(img);
	img4 = image3to4AlphaMap(img, map);
	destroyImage3(&img);
	printStaticImage4(img4);
	destroyImage4(&img4);
}