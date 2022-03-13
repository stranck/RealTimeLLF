#include "imageutils.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

Image makeImage(uint32_t width, uint32_t height){
	Pixel *img = malloc(width * height * sizeof(Pixel)/* * sizeof(uint8_t)*/);
	Image i = {width, height, img};
	return i;
}
Image makeImageWithData(uint32_t width, uint32_t height, Pixel pixels[]){
	size_t dimension = width * height * sizeof(Pixel)/* * sizeof(uint8_t)*/;
	Pixel *img = malloc(dimension);
	memcpy(img, pixels, dimension);
	Image i = {width, height, img};
	return i;
}

void destroyImage(Image img){
	free(img.pixels);
	img.pixels = NULL;
}