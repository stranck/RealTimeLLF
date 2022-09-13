#pragma once

#include <stdint.h>
#include "vects.h"

typedef Vec4f Pixel4;
typedef Vec3f Pixel3;
typedef Vec4u8 Pixel4u8;
typedef Vec3u8 Pixel3u8;
typedef uint8_t* AlphaMap;
typedef float* Kernel;


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