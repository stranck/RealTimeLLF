#pragma once

#include "../utils/imageutils.h"
#include "../utils/extramath.h"
#include "../utils/llfUtils.h"
#include "../utils/structs.h"
#include "../utils/vects.h"
#include "../utils/utils.h"
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

typedef struct {
	Kernel filter;
	Pyramid gaussPyramid;
	Pyramid outputLaplacian;
	Pyramid bufferGaussPyramid;
	Pyramid bufferLaplacianPyramid;
} WorkingBuffers;

void upsample(Image3 *dest, Image3 *source, Kernel filter, Image3 *buffer);
void upsampleConvolve(Image3 *dest, Image3 *source, Kernel kernel);

void laplacianPyramid(Pyramid laplacian, Pyramid tempGauss, uint8_t nLevels, Kernel filter);

void collapse(Image3 *dest, Pyramid laplacianPyr, uint8_t nLevels, Kernel filter);

void downsample(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter, Image3 *buffer);
void downsampleConvolve(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter);

void gaussianPyramid(Pyramid outPyr, Image3 *inImg, uint8_t nLevels, Kernel filter);

void llf(Image3 *img, float sigma, float alpha, float beta, uint8_t nLevels, WorkingBuffers *workingBuffers);
void initWorkingBuffers(WorkingBuffers *workingBuffers, uint32_t width, uint32_t height, uint8_t nLevels);
void destroyWorkingBuffers(WorkingBuffers *workingBuffers, uint8_t nLevels);