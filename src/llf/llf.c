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

void downsample(Image3 *outImg, Image3 *inImg, uint32_t *width, uint32_t *height, Kernel filter, Image3 *buffer){
	convolve(buffer, inImg, filter);
	uint32_t originalW = *width, originalH = *height;
	*width /= 2;
	*height /= 2;
	outImg->width = *width;
	outImg->height = *height;
	uint32_t y;
	uint32_t startingX = originalW & 1;
	uint32_t startingY = originalH & 1;
	for(y = startingY; y < originalH; y += 2) {
		uint32_t x;
		for(x = startingX; x < originalW; x += 2) {
			setPixel3(outImg, x / 2, y / 2, getPixel3(buffer, x - startingX, y - startingY));
		}
	}
}

void gaussianPyramid(Pyramid outPyr, Image3 *inImg, uint8_t nLevels, Kernel filter, Image3 *buffer){
	imgcpy3(outPyr[0], inImg);
	uint32_t width = inImg->width, height = inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsample(buffer, inImg, &width, &height, filter, outPyr[1]); //outPyr[n] is used as a temp buffer
		imgcpy3(outPyr[1], buffer);
	//}
	for(uint8_t i = 1; i < nLevels; i++){
		downsample(buffer, outPyr[i], &width, &height, filter, outPyr[i + 1]);
		imgcpy3(outPyr[i + 1], buffer);
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

	gaussianPyramid(gaussPyramid, img, nLevels, filter, bufferImg);
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
					Image3 *subregion = subimage(img, base_x, end_x, base_y, end_y); //TODO
					remap(subregion, g0, sigma, alpha, beta);

					gaussian_pyramid(bufferGaussPyramid, subregion, lev + 1, filter, bufferImg);
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