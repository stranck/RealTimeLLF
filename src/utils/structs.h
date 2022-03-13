#pragma once

#include <stdint.h>
#include "vects.h"

/*typedef struct {
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t a;
} Pixel;*/
typedef Vec4f Pixel;

typedef struct {
	uint32_t height;
	uint32_t width;
	Pixel *pixels;
} Image;