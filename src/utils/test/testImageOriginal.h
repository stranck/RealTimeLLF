#ifndef STRUCTS_DEP
	#include "structs.h"
#endif
#ifndef IMAGE_UTILS_DEP
	#include "imageutils.h"
#endif
#ifndef UTILS_DEP
	#include "utils.h"
#endif

#ifndef TEST_IMAGE_DEP
#define TEST_IMAGE_DEP

Image getStaticImage(){
	const uint32_t width = 10, height = 10;
	const uint32_t dim = width * height;
	uint32_t data[dim] = {
		0x10000000
	};
	Pixel pxs[dim];
	for(uint32_t i = 0; i < dim; i++){
		// 1 : out = 255 : in
		uint8_t r = (data[i] >> 24) & 0xff;
		uint8_t g = (data[i] >> 16) & 0xff;
		uint8_t b = (data[i] >> 8) & 0xff;
		uint8_t a = data[i] & 0xff;
		pxs[i] = {r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f};
	}
	Image img = makeImage(width, height, pxs);
	return img;
}

void printStaticImage(Image img){
	Pixel *pxs = img.pixels;
	const uint32_t width = img.width;
	const uint32_t height = img.height;
	for(uint32_t i = 0; i < height; i++)
		printBuffer((uint8_t *) &pxs[width * i], width * sizeof(Pixel));
}

#endif