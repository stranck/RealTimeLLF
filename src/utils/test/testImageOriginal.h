#pragma once

#include "../structs.h"
#include "../vects.h"
#include "../imageutils.h"
#include "../utils.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

Image4 * getStaticImage4(){
	const uint32_t width = 10, height = 10;
	const uint32_t dim = width * height;
	uint32_t data[] = {
		0x10000000
	};
	Pixel4 *pxs = (Pixel4 *) malloc(width * height * sizeof(Pixel4));
	for(uint32_t i = 0; i < dim; i++){
		// 1 : out = 255 : in
		uint8_t r = (data[i] >> 24) & 0xff;
		uint8_t g = (data[i] >> 16) & 0xff;
		uint8_t b = (data[i] >> 8) & 0xff;
		uint8_t a = data[i] & 0xff;
		Pixel4 p = {r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f};
		pxs[i] = p;
	}
	Image4 *img = makeImage4WithDataPtr(width, height, pxs);
	return img;
}

void printStaticImage4(Image4 *img){
	Pixel4 *pxs = img->pixels;
	const uint32_t width = img->width;
	const uint32_t height = img->height;
	printf("STATIC_IMG_DIMENSIONS: %d %d\n", width, height);
	Vec4u8 *buffer = (Vec4u8 *) calloc(width, sizeof(Vec4u8));
	for(uint32_t i = 0; i < height; i++){
		Pixel4 *currentLine = &pxs[width * i];
		for(uint32_t j = 0; j < width; j++){
			buffer[j].x = roundfu8(255.0f * currentLine[j].x);
			buffer[j].y = roundfu8(255.0f * currentLine[j].y);
			buffer[j].z = roundfu8(255.0f * currentLine[j].z);
			buffer[j].w = roundfu8(255.0f * currentLine[j].w);
		}
		printBuffer((uint8_t *) buffer, width * sizeof(Vec4u8));
		puts("");
	}
}