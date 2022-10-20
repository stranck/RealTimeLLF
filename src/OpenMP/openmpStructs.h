#pragma once

#include <stdint.h>
#include "../utils/structs.h"

/**
 * @brief Struct containing infos about the current gaussian level we're rendering + cached data
 */
typedef struct {
	uint8_t currentNLevels; //Number of levels in the buffer gaussian pyramid
	uint8_t lev; //Current level
	uint32_t oldY; //Height of the previous level
	uint32_t width; //Width of the current level
	uint32_t prevLevelDimension; //Total number of pixels we worked on the previous frames
	uint32_t nextLevelDimension; //Total number of pixels that will be rendered at the end of this layer. This is used to check when we're done with it
	uint32_t subregionDimension; //Dimension of the rendering subregion at the current level
	uint32_t full_res_roi_yShifted; //Current full_res_roi_yShifted at the current level at the current y
	uint32_t base_y; //Current base_y at the current level at the current y
	uint32_t end_y; //Current end_y at the current level at the current y
	Image3 *currentGaussLevel; //Ptr to the image of the current gaussian level
} CurrentLevelInfo;

/**
 * @brief Struct containing the buffers used in the openmp multithreaded llf rendering
 */
typedef struct {
	uint32_t end; //Total number of pixel we have to render
	Kernel filter; //Blur filter
	Pyramid *bArr; //set of pyramids used by each thread to render each single pixel
	uint32_t *pyrDimensions; //Cached dimensions in pixels for each layer layer of the input gaussian pyramid
	CurrentLevelInfo *cliArr; //Rendering info per each thread
	Pyramid gaussPyramid; //input gaussian pyramid
	Pyramid outputLaplacian; //output laplacian pyramid to be collapsed
} WorkingBuffers;