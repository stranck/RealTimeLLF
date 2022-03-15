#pragma once

#include "../utils/structs.h"

void remap(Image3 * img, Pixel3 g0, double sigma, double alpha, double beta);

#define KERNEL_DIMENSION 5
Kernel createFilter();

Pyramid createPyramid(uint32_t width, uint32_t height, uint8_t nLevels);