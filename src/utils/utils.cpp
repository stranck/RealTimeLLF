#include "utils.h"

void printBuffer(uint8_t *data, uint32_t len){
	uint64_t value;
	uint32_t i = 0, x;
	while(i < len){
		value = 0; //I think this is useless but idk
		for(x = 0; x < 8 && i < len; x++){
			value = value << 8;
			value |= data[i++];
		}
		while(x < 8){
			value = value << 8;
			x++;
		}
		if(i % 8 == 0) {
			#if ON_WINDOWS
				printf("%016I64x", value);
			#else
				printf("%016lx", value);
			#endif
		} else {
			for(x = 0; x < i % 8; x++){
				printf("%02lx", value >> 56 & 0xff);
				value = value << 8;
			}
			i += 2;
		}
	}
}

int32_t roundfI32(float f){
	return (int32_t) roundf(f);
}
uint8_t roundfu8(float f){
	return (uint8_t) roundf(f);
}

