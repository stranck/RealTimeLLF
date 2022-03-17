#include "imageutils.h"
#include "extramath.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

Image4 * makeImage4(uint32_t width, uint32_t height){
	Pixel4 *img = malloc(width * height * sizeof(Pixel4));
	Image4 *i = malloc(sizeof(Image4));
	i -> width = width;
	i -> height = height;
	i -> pixels = img;
	return i;
}
Image4 * makeImage4WithData(uint32_t width, uint32_t height, Pixel4 pixels[]){
	size_t dimension = width * height * sizeof(Pixel4);
	Pixel4 *img = malloc(dimension);
	memcpy(img, pixels, dimension);
	Image4 *i = malloc(sizeof(Image4));
	i -> width = width;
	i -> height = height;
	i -> pixels = img;
	return i;
}
Image4 * makeImage4WithDataPtr(uint32_t width, uint32_t height, Pixel4 *pixels){
	size_t dimension = width * height * sizeof(Pixel4);
	Image4 *i = malloc(sizeof(Image4));
	i -> width = width;
	i -> height = height;
	i -> pixels = pixels;
	return i;
}
Image3 * makeImage3(uint32_t width, uint32_t height){
	Pixel3 *img = malloc(width * height * sizeof(Pixel3));
	Image3 *i = malloc(sizeof(Image3));
	i -> width = width;
	i -> height = height;
	i -> pixels = img;
	return i;
}
Image3 * makeImage3WithData(uint32_t width, uint32_t height, Pixel3 pixels[]){
	size_t dimension = width * height * sizeof(Pixel3);
	Pixel3 *img = malloc(dimension);
	memcpy(img, pixels, dimension);
	Image3 *i = malloc(sizeof(Image3));
	i -> width = width;
	i -> height = height;
	i -> pixels = img;
	return i;
}


void destroyImage3(Image3 **img){
	Image3 *localImg = *img;
	free(localImg -> pixels);
	localImg -> pixels = NULL;
	free(localImg);
	*img = NULL;
}
void destroyImage4(Image4 **img){
	Image4 *localImg = *img;
	free(localImg -> pixels);
	localImg -> pixels = NULL;
	free(localImg);
	*img = NULL;
}

AlphaMap getAlphaMap(Image4 *img){
	uint32_t dimension = img->width * img->height;
	AlphaMap map = malloc(dimension * sizeof(uint8_t));
	for(uint32_t i = 0; i < dimension; i++)
		map[i] = img->pixels[i].w;
	return map;
}
Image3 * image4to3(Image4 *img){
	Image3 *ret = makeImage3(img -> width, img -> height);
	uint32_t dimension = img->width * img->height;
	for(uint32_t i = 0; i < dimension; i++){
		ret->pixels[i].x = img->pixels[i].x;
		ret->pixels[i].y = img->pixels[i].y;
		ret->pixels[i].z = img->pixels[i].z;
	}
	return ret;
}
Image4 * image3to4FixedAlpha(Image3 *img, double alpha){
	Image4 *ret = makeImage4(img -> width, img -> height);
	uint32_t dimension = img->width * img->height;
	for(uint32_t i = 0; i < dimension; i++){
		ret->pixels[i].x = img->pixels[i].x;
		ret->pixels[i].y = img->pixels[i].y;
		ret->pixels[i].z = img->pixels[i].z;
		ret->pixels[i].w = alpha;
	}
	return ret;
}
Image4 * image3to4AlphaMap(Image3 *img, AlphaMap alphaMap){
	Image4 *ret = makeImage4(img -> width, img -> height);
	uint32_t dimension = img->width * img->height;
	for(uint32_t i = 0; i < dimension; i++){
		ret->pixels[i].x = img->pixels[i].x;
		ret->pixels[i].y = img->pixels[i].y;
		ret->pixels[i].z = img->pixels[i].z;
		ret->pixels[i].w = alphaMap[i];
	}
	return ret;
}

void imgcpy3(Image3 *dest, Image3 *source){
	dest->width = source->width;
	dest->height = source->height;
	memcpy(dest->pixels, source->pixels, dest->width * dest->height * sizeof(Pixel3));
}

void subimage3(Image3 *dest, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY){
	uint32_t w = endX - startX;
	uint32_t h = endY - startY;
	dest->width = w;
	dest->height = h;
	for(uint32_t y = 0; y < h; y++){
		uint32_t finalY = startY + y;
		for(uint32_t x = 0; x < w; x++){
			setPixel3(dest, x, y, getPixel3(source, startX + x, finalY));
		}
	}
}

void fillWithColor(Image3 *dest, Pixel3 *color){
	uint32_t dim = dest->width * dest->height;
	Pixel3 *pxs = dest->pixels;
	for(uint32_t i = 0; i < dim; i++)
		pxs[i] = *color;
}

void clampImage3(Image3 *img){
	uint32_t dim = img->width * img->height;
	Pixel3 *px = img->pixels;
	for(uint32_t i = 0; i < dim; i++){
		px[i].x = clamp(px[i].x, 0, 1);
		px[i].y = clamp(px[i].y, 0, 1);
		px[i].z = clamp(px[i].z, 0, 1);
	}
}