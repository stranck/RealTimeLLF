#include "imageutils.h"
#include "structs.h"

Image makeImage(uint32_t width, uint32_t height){
	Pixel *img = malloc(width * height * sizeof(Pixel)/* * sizeof(uint8_t)*/);
	Image i = {width, height, img};
	return i;
}
Image makeImage(uint32_t width, uint32_t height, Pixel pixels[]){
	size_t dimension = width * height * sizeof(Pixel)/* * sizeof(uint8_t)*/;
	Pixel *img = malloc(dimension);
	memcpy(img, pixels, dimension);
	Image i = {width, height, img};
	return i;
}

inline Pixel * getPixel(Image img, uint32_t x, uint32_t y){
	return &img.pixels[y * img.width + x];
}
inline Pixel * getPixel(Image img, Vec2u32 v){
	return &img.pixels[v.y * img.width + v.x];
}

void destroyImage(Image img){
	free(img.pixels);
	img.pixels = NULL;
}