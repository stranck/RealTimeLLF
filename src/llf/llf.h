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

/**
 * @brief Struct containing the buffers used in the singlethread llf rendering
 */
typedef struct {
	Kernel filter; //Blur filter
	Pyramid gaussPyramid; //input gaussian pyramid
	Pyramid outputLaplacian; //output laplacian pyramid to be collapsed
	Pyramid bufferGaussPyramid; //gaussian pyramid using for rendering each single pixel
	Pyramid bufferLaplacianPyramid; //laplacian pyramid using for rendering each single pixel
} WorkingBuffers;

void upsample(Image3 *dest, Image3 *source, Kernel filter, Image3 *buffer);

void collapse(Image3 *dest, Pyramid laplacianPyr, uint8_t nLevels, Kernel filter);

void downsample(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter, Image3 *buffer);

void llf(Image3 *img, float sigma, float alpha, float beta, uint8_t nLevels, WorkingBuffers *workingBuffers);
void initWorkingBuffers(WorkingBuffers *workingBuffers, uint32_t width, uint32_t height, uint8_t nLevels);
void destroyWorkingBuffers(WorkingBuffers *workingBuffers, uint8_t nLevels);