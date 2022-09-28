#pragma once

#include "../utils/imageutils.h"
#include "../utils/extramath.h"
#include "../utils/llfUtils.h"
#include "../utils/structs.h"
#include "../utils/vects.h"
#include "../utils/utils.h"
#include "openmpStructs.h"
#include "openmpUtils.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

Pixel3 upsampleConvolveSubtractSinglePixel(Image3 *source, Pixel3 *gaussPx, Kernel kernel, uint32_t i, uint32_t j);
void upsampleConvolve_parallel(Image3 *dest, Image3 *source, Kernel kernel, const uint8_t nThreads);

void collapse(Image3 *dest, Pyramid laplacianPyr, uint8_t nLevels, Kernel filter, const uint8_t nThreads);

void downsampleConvolve_parallel(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter, const uint8_t nThreads);

void gaussianPyramid_fast(Pyramid outPyr, uint8_t nLevels, Kernel filter);
void gaussianPyramid_parallel(Pyramid outPyr, Image3 *inImg, uint8_t nLevels, Kernel filter, const uint8_t nThreads);

void llf(Image3 *img, float sigma, float alpha, float beta, uint8_t nLevels, const uint8_t nThreads, WorkingBuffers *workingBuffers);
void initWorkingBuffers(WorkingBuffers *workingBuffers, uint32_t width, uint32_t height, uint8_t nLevels, uint8_t nThreads);
void destroyWorkingBuffers(WorkingBuffers *workingBuffers, uint8_t nLevels, uint8_t nThreads);