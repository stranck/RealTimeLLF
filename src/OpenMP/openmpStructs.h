#pragma once

#include <stdint.h>
#include "../utils/structs.h"


typedef struct {
	Pyramid bufferGaussPyramid;
	Pyramid bufferLaplacianPyramid;
	//int ompId;
} Buffers;

#define createBuffers(width, height, nLevels)({\
	Buffers b;\
	b.bufferGaussPyramid = createPyramid(width, height, nLevels);\
	b.bufferLaplacianPyramid = createPyramid(width, height, nLevels);\
	b;\
})

typedef struct {
	uint8_t currentNLevels;
	uint8_t lev;
	uint32_t oldY;
	uint32_t width;
	uint32_t prevLevelDimension;
	uint32_t nextLevelDimension;
	uint32_t subregionDimension;
	uint32_t subregionDimensionPlus1;
	uint32_t full_res_roi_yShifted;
	uint32_t shiftedLev;
	uint32_t base_y;
	uint32_t end_y;
	Image3 *currentGaussLevel;
} CurrentLevelInfo;

