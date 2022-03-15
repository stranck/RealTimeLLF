#pragma once

#include <stdint.h>
#include "vects.h"

/*typedef struct {
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t a;
} Pixel;*/
typedef Vec4f Pixel4;
typedef Vec3f Pixel3;
typedef uint8_t* AlphaMap;
typedef double** Kernel;


typedef struct {
	uint32_t height;
	uint32_t width;
	Pixel4 *pixels;
} Image4;
typedef struct {
	uint32_t height;
	uint32_t width;
	Pixel3 *pixels;
} Image3;

typedef Image3** Pyramid;