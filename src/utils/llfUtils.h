#pragma once

#include "../utils/structs.h"

void remap(Image3 * img, Pixel3 g0, double sigma, double alpha, double beta);

#define KERNEL_DIMENSION 5
#define getKernelPosition(x, y)({x * KERNEL_DIMENSION + y;})
Kernel createFilter();
void destroyFilter(Kernel *filter);

Pyramid createPyramid(uint32_t width, uint32_t height, uint8_t nLevels);
void destroyPyramid(Pyramid *p, uint8_t nLevels);

void convolve(Image3 *dest, Image3 *source, Kernel kernel);