#include "utils.h"
#include <stdio.h>
#include <math.h>

void printBuffer(uint8_t *data, uint32_t len){
	uint64_t value;
	uint32_t i = 0, x;
	while(i < len){
		value = 0; //I think this is useless but idk
		for(x = 0; x < 8 && i < len; x++){
			value = value << 8;
			value |= data[i++];
		}
		printf("%016lx", value);
	}
}

int32_t roundfI32(float f){
	return (int32_t) rintf(f);
}
uint8_t roundfu8(float f){
	return (uint8_t) rintf(f);
}

