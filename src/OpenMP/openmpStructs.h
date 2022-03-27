#pragma once

#include <stdint.h>
#include "../utils/structs.h"


typedef struct {
	Pyramid bufferGaussPyramid;
	Pyramid bufferLaplacianPyramid;
} Buffers;

#define createBuffers(width, height, nLevels)({\
	Buffers b;\
	b.bufferGaussPyramid = createPyramid(width, height, nLevels);\
	b.bufferLaplacianPyramid = createPyramid(width, height, nLevels);\
	b;\
})

typedef struct {
	uint8_t lev;
	uint32_t width;
	uint32_t prevLevelDimension;
	uint32_t nextLevelDimension;
	Image3 *currentGaussLevel;
} CurrentLevelInfo;

