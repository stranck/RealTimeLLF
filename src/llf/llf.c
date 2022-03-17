#include "../utils/imageutils.h"
#include "../utils/extramath.h"
#include "../utils/llfUtils.h"
#include "../utils/structs.h"
#include "../utils/vects.h"
#include "../utils/utils.h"
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#include "../utils/test/testimage.h"

void downsample(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter, Image3 *buffer){
	convolve(buffer, source, filter);
	uint32_t originalW = *width, originalH = *height;
	*width /= 2;
	*height /= 2;
	dest->width = *width;
	dest->height = *height;
	uint32_t y;
	uint32_t startingX = originalW & 1;
	uint32_t startingY = originalH & 1;
	for(y = startingY; y < originalH; y += 2) {
		uint32_t x;
		for(x = startingX; x < originalW; x += 2) {
			setPixel3(dest, x / 2, y / 2, getPixel3(buffer, x - startingX, y - startingY));
		}
	}
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
	//printff("+++ Dest: %dx%d\t\t Allc: %dx%d\t\t Sorc: %dx%d\n", dest->width, dest->height, dest->allocatedW, dest->height, source->width, source->height);

	/*for (uint32_t j = startingY; j < originalH; j += 2) {
		for (uint32_t i = startingX; i < originalW; i += 2) {
			Pixel3 c = zero3f;
			for(uint32_t y = 0; y < rows; y++){
				int32_t jy = (j + ystart + y) * 2;
			}
		}
	}*/


	for (uint32_t j = startingY; j < originalH; j += 2) {
		for (uint32_t i = startingX; i < originalW; i += 2) {
			Pixel3 c = zero3f;
			for (uint32_t y = 0; y < rows; y++) {
				int32_t jy = j + (ystart + y) * 2 - startingY;
				for (uint32_t x = 0; x < cols; x++) {
					int32_t ix = i + (xstart + x) * 2 - startingX;
					//if(j == (166 * 2) && i == (148 * 2)) printff("PX pre: 0x%02x 0x%02x 0x%02x\n", roundfu8(255 * c.x), roundfu8(255 * c.y), roundfu8(255 * c.z));

					if (ix >= 0 && ix < originalW && jy >= 0 && jy < originalH) {
						
						double kern_elem = filter[getKernelPosition(x, y)];
						//if(j == (166 * 2) && i == (148 * 2)) printff("Getting pixel #1: %d ; %d (Org: %d ; %d) (xy:  %d ; %d) (startXY: %d ; %d)\n", ix, jy, ix + startingX, jy + startingY, x, y, xstart, ystart);
						Pixel3 px = *getPixel3(source, ix, jy);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					} else {
						
						double kern_elem = filter[getKernelPosition(x, y)];
						//if(j == (166 * 2) && i == (148 * 2)) printff("Getting pixel #2: %d ; %d (Org: %d ; %d)\n", i - startingX, j - startingY, i, j);
						Pixel3 px = *getPixel3(source, i - startingX, j - startingY);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					}
					//if(j == (166 * 2) && i == (148 * 2)) printff("PX pst: 0x%02x 0x%02x 0x%02x\n", roundfu8(255 * c.x), roundfu8(255 * c.y), roundfu8(255 * c.z));
				}
			}
			//if(getPixel3(dest, i / 2, j / 2)->y == 0) 
			//if(j == (166 * 2) && i == (148 * 2)) printff("Setting pixel: %d ; %d (Org: %d ; %d)\n", i / 2, j / 2, i, j);
			setPixel3(dest, i / 2, j / 2, &c);
		}
	}
}

void upsample(Image3 *dest, Image3 *source, Kernel filter, Image3 *buffer){
	uint32_t smallWidth = source->width, smallHeight = source->height;
	uint32_t uppedW = smallWidth << 1;
	uint32_t uppedH = smallHeight << 1;
	buffer->width = uppedW;
	buffer->height = uppedH;
	for(uint32_t y = 0; y < smallHeight; y++){
		uint32_t yUp = y * 2;
		uint32_t yUpLess = yUp++;
		for(uint32_t x = 0; x < smallWidth; x++){
			uint32_t xUp = x * 2;
			Pixel3 *pixel = getPixel3(source, x, y);
			uint32_t xUpLess = xUp++;

			setPixel3(buffer, xUpLess, yUpLess, pixel);
			setPixel3(buffer, xUpLess, yUp, pixel);
			setPixel3(buffer, xUp, yUpLess, pixel);
			setPixel3(buffer, xUp, yUp, pixel);
		}
	}
	convolve(dest, buffer, filter);
}
void upsampleConvolve(Image3 *dest, Image3 *source, Kernel kernel){
	//printff("Source addr: 0x%016lx    Pxs addr: 0x%016lx\n", source, source->pixels);
	uint32_t smallWidth = source->width, smallHeight = source->height;
	uint32_t uppedW = smallWidth << 1;
	uint32_t uppedH = smallHeight << 1;
	dest->width = uppedW;
	dest->height = uppedH;
	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;
	//printff("Small: %dx%d\t\t Big: %dx%d", uppedW, smallHeight, smallWidth, smallHeight);

	for (uint32_t j = 0; j < uppedH; j++) {
		for (uint32_t i = 0; i < uppedW; i++) {
			Pixel3 c = zero3f;
			for (uint32_t y = 0; y < rows; y++) {
                int32_t jy = (j + ystart + y) / 2;
				for (uint32_t x = 0; x < cols; x++) {
                    int32_t ix = (i + xstart + x) / 2;
                    if (ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight) {
						double kern_elem = kernel[getKernelPosition(x, y)];
						//printff("Getting pixel #1: %d ; %d (Org: %d ; %d) (xy:  %d ; %d) (startXY: %d ; %d)\n", ix / 2, jy / 2, ix, jy, x, y, xstart, ystart);
						Pixel3 px = *getPixel3(source, ix, jy);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					} else {
						double kern_elem = kernel[getKernelPosition(x, y)];
						//printff("Getting pixel #2: %d ; %d (Org: %d ; %d)\n", i / 2, j / 2, i, j);
						Pixel3 px = *getPixel3(source, i / 2, j / 2);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					}
				}
			}
			//printff("Setting pixel: %d ; %d (Coming from: %d ; %d)\n", i, j, i / 2, j / 2);
			setPixel3(dest, i, j, &c);
		}
	}
}

void gaussianPyramid(Pyramid outPyr, Image3 *inImg, uint8_t nLevels, Kernel filter){
	//print("Copying img");
	//printff("outPyr[0]: %dx%d\t\t inImg: %dx%d\n", outPyr[0]->width, outPyr[0]->height, inImg->width, inImg->height);
	imgcpy3(outPyr[0], inImg);
	//print("Copying img DONE");
	uint32_t width = inImg->width, height = inImg->height;
	//print("Downsampling");
	//printff("outPyr[1]: %dx%d\t\t inImg: %dx%d\n", outPyr[1]->width, outPyr[1]->height, inImg->width / 2, inImg->height / 2);
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve(outPyr[1], inImg, &width, &height, filter);
	//print("Downsampling DONE");
	//}
	for(uint8_t i = 1; i < nLevels; i++){
		//printf("%d / %d\n", i, nLevels); fflush(stdout);
		//printff("outPyr[i + 1]: %dx%d\t\t outPyr[i]: %dx%d\n", outPyr[i + 1]->width, outPyr[i + 1]->height, outPyr[i]->width, outPyr[i]->height);
		downsampleConvolve(outPyr[i + 1], outPyr[i], &width, &height, filter);
		//print("downsample done");
	}
	//print("Exiting");
}

void laplacianPyramid(Pyramid laplacian, Pyramid tempGauss, uint8_t nLevels, Kernel filter){
	for(uint8_t i = 0; i < nLevels; i++){ //Not sure about the -1
		Image3 *upsampled = laplacian[i];
		upsampleConvolve(upsampled, tempGauss[i + 1], filter);

		Image3 *current = tempGauss[i];
		uint32_t yEnd = min(current->height, upsampled->height);
		uint32_t xEnd = min(current->width, upsampled->width);
		for (uint32_t y = 0; y < yEnd; y++){
			for (uint32_t x = 0; x < xEnd; x++){
				Pixel3 *upsPtr = getPixel3(upsampled, x, y);
				Pixel3 ups = *upsPtr;
				Pixel3 crr = *getPixel3(current, x, y);

				//printff("RES: (%f %f %f)   CRR: (%f %f %f)   UPS: (%f %f %f)\n", (crr.x - upsPtr->x), (crr.y - upsPtr->y), (crr.z - upsPtr->z), crr.x, crr.y, crr.z, upsPtr->x, upsPtr->y, upsPtr->z);
				//printff("RES: (0x%02x 0x%02x 0x%02x)   CRR: (0x%02x 0x%02x 0x%02x)   UPS: (0x%02x 0x%02x 0x%02x)\n", roundfu8(255 * (crr.x - upsPtr->x)), roundfu8(255 * (crr.y - upsPtr->y)), roundfu8(255 * (crr.z - upsPtr->z)), roundfu8(255 * crr.x), roundfu8(255 * crr.y), roundfu8(255 * crr.z), roundfu8(255 * upsPtr->x), roundfu8(255 * upsPtr->y), roundfu8(255 * upsPtr->z));
				*upsPtr = vec3Sub(crr, ups, Pixel3);
				//upsPtr->x = clamp(crr.x - upsPtr->x, 0, 1);
				//upsPtr->y = clamp(crr.y - upsPtr->y, 0, 1);
				//upsPtr->z = clamp(crr.z - upsPtr->z, 0, 1);
			}
		}
	}
	imgcpy3(laplacian[nLevels], tempGauss[nLevels]);
}

/*
//Pseudocode bc this shit keeps segfaulting
void collapse(dest, pyramid, levels){
	upsample(pyr[2], pyr[3], pyr[2]);
	upsample(pyr[1], pyr[2], pyr[1]);
	upsample(dest, pyr[1], pyr[0]);
} 
void collapse(dest, pyramid, levels){
	upsample(pyr[2], pyr[3], pyr[2]);{
		upsample(dest, pyr[3])
		pyr[2] = dest + pyr[2]
	}
	upsample(pyr[1], pyr[2], pyr[1]);
	upsample(dest, pyr[1], pyr[0]){
		upsample(dest, pyr[1]);
		dest = dest + pyr[0]
	}
} 
 */
void collapse(Image3 *dest, Pyramid laplacianPyr, uint8_t nLevels, Kernel filter){
	Pixel3 *destPxs = dest->pixels;
	for(int8_t lev = nLevels; lev > 1; lev--){ //Using dest as a temp buffer
		printff("%d / 0\n", lev);	
		Image3 *currentLevel = laplacianPyr[lev], *biggerLevel = laplacianPyr[lev - 1];
		Pixel3 *biggerLevelPxs = biggerLevel->pixels;

		upsampleConvolve(dest, currentLevel, filter);
		print("Returned from upsample");
		uint32_t sizeUpsampled = min(dest->width, biggerLevel->width) * min(dest->height, biggerLevel->height);
		for(uint32_t px = 0; px < sizeUpsampled; px++)
			biggerLevelPxs[px] = vec3Add(destPxs[px], biggerLevelPxs[px], Pixel3);
		biggerLevel->width = dest->width;
		biggerLevel->height = dest->height; //This could cause disalignment problem
	}
	print("Out of collapse loop");
	//We save one extra copy by doing the last collapse directly inside dest
	Image3 *currentLevel = laplacianPyr[1], *biggerLevel = laplacianPyr[0];
	Pixel3 *biggerLevelPxs = biggerLevel->pixels;

	upsampleConvolve(dest, currentLevel, filter);
	print("Returned from upsample");
	uint32_t sizeUpsampled = min(dest->width, biggerLevel->width) * min(dest->height, biggerLevel->height);
	for(uint32_t px = 0; px < sizeUpsampled; px++)
		destPxs[px] = vec3Add(destPxs[px], biggerLevelPxs[px], Pixel3);
	print("Returning from collapse");

	/*Image3 *result = laplacianPyr[nLevels];
	Pixel3 *destPxs = dest->pixels, *psxUpsampled = result->pixels;
	/*if(nLevels - 1 >= 0){ //We save one extra copy by using dest as a temp buffer
		Image3 *pyr = laplacianPyr[nLevels - 1];
		uint32_t pyrWidth = pyr->width, pyrHeight = pyr->height;
		Pixel3 *psxPyr = pyr->pixels;

		upsampleConvolve(dest, result, filter);
		uint32_t sizeUpsampled = min(dest->width, pyrWidth) * min(dest->height, pyrHeight);
		for(uint32_t px = 0; px < sizeUpsampled; px++)
			psxUpsampled[px] = vec3Add(psxPyr[px], psxUpsampled[px], Pixel3);
		result->width = dest->width;
		result->height = dest->height;
	}*//*
	printff("Remaining levels: %d\n", nLevels - 2);	
	for(int8_t lev = nLevels - 1; lev >= 0; lev--){
		//printff("#1 DEST    Source addr: 0x%016lx    Pxs addr: 0x%016lx\n", dest, dest->pixels);
		//for(int i = 0; i <= nLevels; i++) printff("#1 LAPL[%d] Source addr: 0x%016lx    Pxs addr: 0x%016lx\n", i, laplacianPyr[i], laplacianPyr[i]->pixels);
		//printff("%d / 0\n", lev);	
		Image3 *pyr = laplacianPyr[lev];
		uint32_t pyrWidth = pyr->width, pyrHeight = pyr->height;
		Pixel3 *psxPyr = pyr->pixels;

		//printff("+++ Dest: %dx%d\t\t Allc: %dx%d\t\t Sorc: %dx%d\n", dest->width, dest->height, dest->allocatedW, dest->height, result->width, result->height);
		upsampleConvolve(dest, result, filter);
		//printff("#2 DEST    Source addr: 0x%016lx    Pxs addr: 0x%016lx\n", dest, dest->pixels);
		//for(int i = 0; i <= nLevels; i++) printff("#2 LAPL[%d] Source addr: 0x%016lx    Pxs addr: 0x%016lx\n", i, laplacianPyr[i], laplacianPyr[i]->pixels);
		//print("Returned from upsample");
		uint32_t sizeUpsampled = min(dest->width, pyrWidth) * min(dest->height, pyrHeight);
		for(uint32_t px = 0; px < sizeUpsampled; px++)
			psxUpsampled[px] = vec3Add(psxPyr[px], destPxs[px], Pixel3);
		result->width = dest->width;
		result->height = dest->height;
	}*/

}

void llf(Image3 *img, double sigma, double alpha, double beta, uint8_t nLevels){
	uint32_t width = img->width, height = img->height;
	nLevels = min(nLevels, 5);
	nLevels = max(nLevels, 3);//int(ceil(std::abs(std::log2(min(width, height)) - 3))) + 2;
	Kernel filter = createFilter();
	Pyramid gaussPyramid = createPyramid(width, height, nLevels);
	Pyramid outputLaplacian = createPyramid(width, height, nLevels);
	Pyramid bufferGaussPyramid = createPyramid(width, height, nLevels);
	Pyramid bufferLaplacianPyramid = createPyramid(width, height, nLevels);

	gaussianPyramid(gaussPyramid, img, nLevels, filter);
	//imgcpy3(img, gaussPyramid[0]);
	//return;
	print("Testing convolve"); //upsampleConvolve(img, gaussPyramid[2], filter);
	print("Entering loop");
	for(uint8_t lev = 0; lev < nLevels; lev++){
		printff("laplacian inner loop %d/%d\n", lev, (nLevels - 1));
		Image3 *currentGaussLevel = gaussPyramid[lev];
		uint32_t gaussianWidth = currentGaussLevel->width, gaussianHeight = currentGaussLevel->height;
		uint32_t subregionDimension = 3 * ((1 << (lev + 2)) - 1) / 2;

		for(uint32_t y = 0; y < gaussianHeight; y++){
			//printff("laplacian inner loop %d/%d\ty = %d/%d\n", lev, (nLevels - 1), y, gaussianHeight);

			//no fuckin clues what this calcs are
			int32_t full_res_y = (1 << lev) * y;
			int32_t roi_y0 = full_res_y - subregionDimension;
			int32_t roi_y1 = full_res_y + subregionDimension + 1;
			int32_t base_y = max(0, roi_y0);
			int32_t end_y = min(roi_y1, height);
			int32_t full_res_roi_y = full_res_y - base_y;
			int32_t full_res_roi_yShifted = full_res_roi_y >> lev;

			for(uint32_t x = 0; x < gaussianWidth; x++){
				//no fuckin clues what this calcs are PT2
				int32_t full_res_x = (1 << lev) * x;
				int32_t roi_x0 = full_res_x - subregionDimension;
				int32_t roi_x1 = full_res_x + subregionDimension + 1;
				int32_t base_x = max(0, roi_x0);
				int32_t end_x = min(roi_x1, width);
				int32_t full_res_roi_x = full_res_x - base_x;

				Pixel3 g0 = *getPixel3(currentGaussLevel, x, y);
				subimage3(bufferLaplacianPyramid[0], img, base_x, end_x, base_y, end_y); //Using bufferLaplacianPyramid[0] as temp buffer
				remap(bufferLaplacianPyramid[0], g0, sigma, alpha, beta);

				uint8_t currentNLevels = lev + 1;
				gaussianPyramid(bufferGaussPyramid, bufferLaplacianPyramid[0], currentNLevels, filter);
				laplacianPyramid(bufferLaplacianPyramid, bufferGaussPyramid, currentNLevels, filter);

				setPixel3(outputLaplacian[lev], x, y, getPixel3(bufferLaplacianPyramid[lev], full_res_roi_x >> lev, full_res_roi_yShifted)); //idk why i had to shift those
			}
		}
	}
	imgcpy3(outputLaplacian[nLevels], gaussPyramid[nLevels]);
	print("Exiting loop");
	collapse(img, outputLaplacian, nLevels, filter);
	
	print("Destorying stuff PEW PEW");
	/*destroyPyramid(&gaussPyramid, nLevels);
	destroyPyramid(&outputLaplacian, nLevels);
	destroyPyramid(&bufferGaussPyramid, nLevels);
	destroyPyramid(&bufferLaplacianPyramid, nLevels);
	destroyFilter(&filter);*/
}

int main(){
	print("\n\n\n  OUTPUT  \nvvvvvvvvvv");
	Image4 *img4 = getStaticImage4();
	//print("ASDDD");
	//uint8_t buffer[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10};
	//printBuffer(&buffer, sizeof(buffer));
	//print("Done");
	Image3 *img = image4to3(img4);
	AlphaMap map = getAlphaMap(img4);

	Pyramid bufferGaussPyramid = createPyramid(img->width, img->height, 3);
	Pyramid bufferLaplacianPyramid = createPyramid(img->width, img->height, 3);

	//destroyImage(img4);
	llf(img, 0.35, 0.4, 5, 3);
	//Kernel filter = createFilter();
	//Image3 *cpy = makeImage3(img->width, img->height);
	//Pixel3 px = {1, 0, 0};
	//fillWithColor(bufferLaplacianPyramid[2], &px);
	//Pixel3 px = {0x7f, 0x7f, 0x7f};
	//remap(img, px, 0.35, 0.4, 5);
	//gaussianPyramid(bufferGaussPyramid, img, 3, filter);
	//laplacianPyramid(bufferLaplacianPyramid, bufferGaussPyramid, 3, filter);
	//collapse(img, bufferLaplacianPyramid, 3, filter);

	//imgcpy3(img, bufferLaplacianPyramid[2]);


	//subimage3(img, cpy, 256, 396, 42, 212);
	clampImage3(img);
	img4 = image3to4AlphaMap(img, map);
	//destroyImage(img);
	//printff("%d x %d\n", img4->width, img4->height);
	//printStaticImage4(img4);
	//	destroyImage(img4);
}