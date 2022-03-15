#include "../utils/imageutils.h"
#include "../utils/llfUtils.h"
#include "../utils/structs.h"
#include "../utils/vects.h"
#include "../utils/utils.h"
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#include "../utils/test/testimage.h"


/*
Image4 upsample(Image4 I, double filter[]){
	int smallWidth = I.width, smallHeight = I.height;
	int uppedW = smallWidth << 1;
	int uppedH = smallHeight << 1;
	Image4 upsampled = make_image(uppedW, uppedH, false);
	for(int y = 0; y < smallHeight; y++){
		int yUp = y * 2;
		int yUpLess = yUp++;
		for(int x = 0; x < smallWidth; x++){
			int xUp = x * 2;
			auto pixel = I[{x, y}];
			int xUpLess = xUp++;

			upsampled[{xUpLess, yUpLess}] = pixel;
			upsampled[{xUpLess, yUp}] = pixel;
			upsampled[{xUp, yUpLess}] = pixel;
			upsampled[{xUp, yUp}] = pixel;
		}
	}
	
	return convolve(upsampled, filter);
}*/

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
	uint32_t y;
	const uint32_t startingX = originalW & 1;
	const uint32_t startingY = originalH & 1;
	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;

	for (uint32_t j = 0; j < originalH; j += 2) {
		for (uint32_t i = 0; i < originalW; i += 2) {
			Pixel3 c = zero3f;
			for (uint32_t y = 0; y < rows; y++) {
                int32_t jy = j + ystart + y;
				for (uint32_t x = 0; x < cols; x++) {
                    int32_t ix = i + xstart + x;
                    if (ix >= 0 && ix < dest->width && jy >= 0 && jy < dest->height) {
						double kern_elem = filter[x][y];
						Pixel3 px = *getPixel3(source, ix - startingX, jy - startingY);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					} else {
						double kern_elem = filter[x][y];
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

void gaussianPyramid(Pyramid outPyr, Image3 *inImg, uint8_t nLevels, Kernel filter){
	imgcpy3(outPyr[0], inImg);
	uint32_t width = inImg->width, height = inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve(outPyr[1], inImg, &width, &height, filter);
	//}
	for(uint8_t i = 1; i < nLevels; i++){
		downsampleConvolve(outPyr[i + 1], outPyr[i], &width, &height, filter);
	}
}

Image3 * llf(Image3 *img, double sigma, double alpha, double beta, uint8_t nLevels){
	uint32_t width = img->width, height = img->height;
	nLevels = max(min(nLevels, 5), 3);//int(ceil(std::abs(std::log2(min(width, height)) - 3))) + 2;
	Kernel filter = createFilter();
	Pyramid gaussPyramid = createPyramid(width, height, nLevels);
	Pyramid outputLaplacian = createPyramid(width, height, nLevels);

	Pyramid bufferGaussPyramid = createPyramid(width, height, nLevels);
	Pyramid bufferLaplacianPyramid = createPyramid(width, height, nLevels);
	Image3 *bufferImg = makeImage3(width, height);

	gaussianPyramid(gaussPyramid, img, nLevels, filter);
	for(uint8_t lev = 0; lev < nLevels; lev++){
			Image3 *currentGaussLevel = gaussPyramid[lev];
			uint32_t gaussianWidth = currentGaussLevel->width, gaussianHeight = currentGaussLevel->height;
			uint32_t subregionDimension = 3 * ((1 << (lev + 2)) - 1) / 2;

			for(uint32_t y = 0; y < gaussianHeight; y++){

				//no fuckin clues what this calcs are
				uint32_t full_res_y = (1 << lev) * y;
				uint32_t roi_y0 = full_res_y - subregionDimension;
				uint32_t roi_y1 = full_res_y + subregionDimension + 1;
				uint32_t base_y = max(0, roi_y0);
				uint32_t end_y = min(roi_y1, height);
				uint32_t full_res_roi_y = full_res_y - base_y;
				uint32_t full_res_roi_yShifted = full_res_roi_y >> lev;

				for(uint32_t x = 0; x < gaussianWidth; x++){
					//no fuckin clues what this calcs are PT2
					uint32_t full_res_x = (1 << lev) * x;
					uint32_t roi_x0 = full_res_x - subregionDimension;
					uint32_t roi_x1 = full_res_x + subregionDimension + 1;
					uint32_t base_x = max(0, roi_x0);
					uint32_t end_x = min(roi_x1, width);
					uint32_t full_res_roi_x = full_res_x - base_x;

					Pixel3 g0 = *getPixel3(currentGaussLevel, x, y);
					subimage(bufferLaplacianPyramid[0], img, base_x, end_x, base_y, end_y); //Using bufferLaplacianPyramid[0] as temp buffer
					remap(bufferLaplacianPyramid[0], g0, sigma, alpha, beta);

					gaussian_pyramid(bufferGaussPyramid, bufferLaplacianPyramid[0], lev + 1, filter, bufferImg);
					laplacian_pyramid(bufferLaplacianPyramid, bufferGaussPyramid, filter, bufferImg); //TODO

					setPixel(outputLaplacian[lev], x, y, getPixel(bufferLaplacianPyramid[lev], full_res_roi_x >> lev, full_res_roi_yShifted)); //idk why i had to shift those
				}
			}
		}
}

int main(){
	Image4 *img4 = getStaticImage4();
	Image3 *img = image4to3(img4);
	AlphaMap map = getAlphaMap(img4);
	destroyImage(img4);
	Pixel3 test = *getPixel3(img, img->width / 2, img->height / 2);
	remap(img, test, 0.35, 0.4, 5);
	img4 = image3to4AlphaMap(img, map);
	printStaticImage4(img4);
	destroyImage(img4);
}