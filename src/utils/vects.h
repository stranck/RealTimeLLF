#pragma once

#include <stdint.h>

typedef struct {
	int32_t x;
	int32_t y;
} Vec2i32;

typedef struct {
	int32_t x;
	int32_t y;
	int32_t z;
} Vec3i32;

typedef struct {
	int32_t x;
	int32_t y;
	int32_t z;
	int32_t w;
} Vec4i32;


typedef struct {
	uint32_t x;
	uint32_t y;
} Vec2u32;

typedef struct {
	uint32_t x;
	uint32_t y;
	uint32_t z;
} Vec3u32;

typedef struct {
	uint32_t x;
	uint32_t y;
	uint32_t z;
	uint32_t w;
} Vec4u32;


typedef struct {
	uint8_t x;
	uint8_t y;
} Vec2u8;

typedef struct {
	uint8_t x;
	uint8_t y;
	uint8_t z;
} Vec3u8;

typedef struct {
	uint8_t x;
	uint8_t y;
	uint8_t z;
	uint8_t w;
} Vec4u8;


typedef struct {
	float x;
	float y;
} Vec2f;

typedef struct {
	float x;
	float y;
	float z;
} Vec3f;

typedef struct {
	float x;
	float y;
	float z;
	float w;
} Vec4f;

#define vec4Add(dest, v1, v2){\
	(dest).x = (v1).x + (v2).x;\
	(dest).y = (v1).y + (v2).y;\
	(dest).z = (v1).z + (v2).z;\
	(dest).w = (v1).w + (v2).w;\
}
#define vec4Sub(dest, v1, v2){\
	(dest).x = (v1).x - (v2).x;\
	(dest).y = (v1).y - (v2).y;\
	(dest).z = (v1).z - (v2).z;\
	(dest).w = (v1).w - (v2).w;\
}
#define vec4Mul(dest, v1, v2){\
	(dest).x = (v1).x * (v2).x;\
	(dest).y = (v1).y * (v2).y;\
	(dest).z = (v1).z * (v2).z;\
	(dest).w = (v1).w * (v2).w;\
}
#define vec4Div(dest, v1, v2){\
	(dest).x = (v1).x / (v2).x;\
	(dest).y = (v1).y / (v2).y;\
	(dest).z = (v1).z / (v2).z;\
	(dest).w = (v1).w / (v2).w;\
}


#define vec4AddC(dest, v1, c){\
	(dest).x = (v1).x + c;\
	(dest).y = (v1).y + c;\
	(dest).z = (v1).z + c;\
	(dest).w = (v1).w + c;\
}
#define vec4SubC(dest, v1, c){\
	(dest).x = (v1).x - c;\
	(dest).y = (v1).y - c;\
	(dest).z = (v1).z - c;\
	(dest).w = (v1).w - c;\
}
#define vec4MulC(dest, v1, c){\
	(dest).x = (v1).x * c;\
	(dest).y = (v1).y * c;\
	(dest).z = (v1).z * c;\
	(dest).w = (v1).w * c;\
}
#define vec4DivC(dest, v1, c){\
	(dest).x = (v1).x / c;\
	(dest).y = (v1).y / c;\
	(dest).z = (v1).z / c;\
	(dest).w = (v1).w / c;\
}



#define vec3Add(dest, v1, v2){\
	(dest).x = (v1).x + (v2).x;\
	(dest).y = (v1).y + (v2).y;\
	(dest).z = (v1).z + (v2).z;\
}
#define vec3Sub(dest, v1, v2){\
	(dest).x = (v1).x - (v2).x;\
	(dest).y = (v1).y - (v2).y;\
	(dest).z = (v1).z - (v2).z;\
}
#define vec3Mul(dest, v1, v2){\
	(dest).x = (v1).x * (v2).x;\
	(dest).y = (v1).y * (v2).y;\
	(dest).z = (v1).z * (v2).z;\
}
#define vec3Div(dest, v1, v2){\
	(dest).x = (v1).x / (v2).x;\
	(dest).y = (v1).y / (v2).y;\
	(dest).z = (v1).z / (v2).z;\
}


#define vec3AddC(dest, v1, c){\
	(dest).x = (v1).x + c;\
	(dest).y = (v1).y + c;\
	(dest).z = (v1).z + c;\
}
#define vec3SubC(dest, v1, c){\
	(dest).x = (v1).x - c;\
	(dest).y = (v1).y - c;\
	(dest).z = (v1).z - c;\
}
#define vec3MulC(dest, v1, c){\
	(dest).x = (v1).x * c;\
	(dest).y = (v1).y * c;\
	(dest).z = (v1).z * c;\
}
#define vec3DivC(dest, v1, c){\
	(dest).x = (v1).x / c;\
	(dest).y = (v1).y / c;\
	(dest).z = (v1).z / c;\
}

#define zero2vect {0, 0}
#define zero3vect {0, 0, 0}
#define zero4vect {0, 0, 0, 0}

#define one2vect {1, 1}
#define one3vect {1, 1, 1}
#define one4vect {1, 1, 1, 1}