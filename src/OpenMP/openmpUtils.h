#pragma once

#include <omp.h>
#include <stdint.h>
#include "../utils/structs.h"
#include "openmpStructs.h"

void initLevelInfo(CurrentLevelInfo *cli, uint32_t *pyrDimensions, Pyramid gaussPyramid);
void updateLevelInfo(CurrentLevelInfo *cli, uint32_t *pyrDimensions, Pyramid gaussPyramid);

void imgcpy3_parallel(Image3 *dest, Image3 *source, const uint8_t nThreads);
void clampImage3_parallel(Image3 *img, const uint8_t nThreads);

#define getThreadId()(omp_get_thread_num())