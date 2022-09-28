#include "utils.h"

/**
 * @brief Prints in hex a buffer. Big endian.
 * If the buffer is not a multiple of 8 bytes long it appends some 0s
 * It prints 8 bytes per loop, to reduce printf call overhead
 * 
 * @param data Buffer to print
 * @param len Length of the buffer
 */
void printBuffer(uint8_t *data, uint32_t len){
	uint64_t value;
	uint32_t i = 0, x;
	while(i < len){
		value = 0; //I think this is useless but idk
		//big endian load of 8 bytes of the buffer
		for(x = 0; x < 8 && i < len; x++){
			value = value << 8;
			value |= data[i++];
		}
		//If we have read less than 8 bytes, append 0s till we reach 8 bytes
		while(x < 8){
			value = value << 8;
			x++;
		}
		//Print all together the 8 bytes we have just loaded
		#if ON_WINDOWS
			printf("%016I64x", value); //msvc is not happy if you use the lx format string for a uint64_t
		#else
			printf("%016lx", value);
		#endif
	}
}
