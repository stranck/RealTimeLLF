#pragma once

#include <stdint.h>
#include "../structs.h"
#include "openmpStructs.h"

void initLevelInfo(CurrentLevelInfo *cli, uint32_t *pyrDimensions, Pyramid gaussPyramid);
void updateLevelInfo(CurrentLevelInfo *cli, uint32_t *pyrDimensions, Pyramid gaussPyramid);

void imgcpy3_parallel(Image3 *dest, Image3 *source, const uint8_t nThreads);