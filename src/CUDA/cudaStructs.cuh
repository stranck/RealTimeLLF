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

