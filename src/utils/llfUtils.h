#pragma once

#include "imageutils.h"
#include "extramath.h"
#include "structs.h"
#include "vects.h"
#include "utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void remap(Image3 * img, Pixel3 g0, float sigma, float alpha, float beta);

#define KERNEL_DIMENSION 5
/**
 * @brief returns the kernel element located at x, y
 */
#define getKernelPosition(x, y)(x * KERNEL_DIMENSION + y)
Kernel createFilter();
void destroyFilter(Kernel *filter);

Pyramid createPyramid(uint32_t width, uint32_t height, uint8_t nLevels);
void destroyPyramid(Pyramid *p, uint8_t nLevels);

void convolve(Image3 *dest, Image3 *source, Kernel kernel);

void upsampleConvolve(Image3 *dest, Image3 *source, Kernel kernel);
void laplacianPyramid(Pyramid laplacian, Pyramid tempGauss, uint8_t nLevels, Kernel filter);

void downsampleConvolve(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter);
void gaussianPyramid(Pyramid outPyr, Image3 *inImg, uint8_t nLevels, Kernel filter);
