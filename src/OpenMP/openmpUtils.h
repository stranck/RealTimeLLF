#pragma once

#include <stdint.h>
#include "../structs.h"
#include "openmpStructs.h"

void initLevelInfo(CurrentLevelInfo *cli, uint32_t *pyrDimensions, Pyramid gaussPyramid);
void updateLevelInfo(CurrentLevelInfo *cli, uint32_t *pyrDimensions, Pyramid gaussPyramid);