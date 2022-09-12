#pragma once

#include <stdint.h>
#include "../utils/structs.h"


typedef struct {
	Pyramid bufferGaussPyramid;
	Pyramid bufferLaplacianPyramid;
} Buffers;

#define createBuffers(dest, width, height, nLevels){\
	(dest).bufferGaussPyramid = createPyramid(width, height, nLevels);\
	(dest).bufferLaplacianPyramid = createPyramid(width, height, nLevels);\
}

typedef struct {
	uint8_t currentNLevels;
	uint8_t lev;
	uint32_t oldY;
	uint32_t width;
	uint32_t prevLevelDimension;
	uint32_t nextLevelDimension;
	uint32_t subregionDimension;
	uint32_t full_res_roi_yShifted;
	uint32_t base_y;
	uint32_t end_y;
	Image3 *currentGaussLevel;
} CurrentLevelInfo;

