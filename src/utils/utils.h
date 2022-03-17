#pragma once

#include <stdint.h>
#include <math.h>

void printBuffer(uint8_t *data, uint32_t len);

int32_t roundfI32(float f);
uint8_t roundfu8(float f);

#define print(str){puts(str); fflush(stdout);}
#define printff(format, args...){fprintf(stderr, format, args); fflush(stdout);}
