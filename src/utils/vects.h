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

#define vec4Add(v1, v2, t)({\
	t ret;\
	ret.x = v1.x + v2.x;\
	ret.y = v1.y + v2.y;\
	ret.z = v1.z + v2.z;\
	ret.w = v1.w + v2.w;\
	ret; \
})
#define vec4Sub(v1, v2, t)({\
	t ret;\
	ret.x = v1.x - v2.x;\
	ret.y = v1.y - v2.y;\
	ret.z = v1.z - v2.z;\
	ret.w = v1.w - v2.w;\
	ret; \
})
#define vec4Mul(v1, v2, t)({\
	t ret;\
	ret.x = v1.x * v2.x;\
	ret.y = v1.y * v2.y;\
	ret.z = v1.z * v2.z;\
	ret.w = v1.w * v2.w;\
	ret; \
})
#define vec4Div(v1, v2, t)({\
	t ret;\
	ret.x = v1.x / v2.x;\
	ret.y = v1.y / v2.y;\
	ret.z = v1.z / v2.z;\
	ret.w = v1.w / v2.w;\
	ret; \
})


#define vec4AddC(v1, c, t)({\
	t ret;\
	ret.x = v1.x + c;\
	ret.y = v1.y + c;\
	ret.z = v1.z + c;\
	ret.w = v1.w + c;\
	ret; \
})
#define vec4SubC(v1, c, t)({\
	t ret;\
	ret.x = v1.x - c;\
	ret.y = v1.y - c;\
	ret.z = v1.z - c;\
	ret.w = v1.w - c;\
	ret; \
})
#define vec4MulC(v1, c, t)({\
	t ret;\
	ret.x = v1.x * c;\
	ret.y = v1.y * c;\
	ret.z = v1.z * c;\
	ret.w = v1.w * c;\
	ret; \
})
#define vec4DivC(v1, c, t)({\
	t ret;\
	ret.x = v1.x / c;\
	ret.y = v1.y / c;\
	ret.z = v1.z / c;\
	ret.w = v1.w / c;\
	ret; \
})



#define vec3Add(v1, v2, t)({\
	t ret;\
	ret.x = v1.x + v2.x;\
	ret.y = v1.y + v2.y;\
	ret.z = v1.z + v2.z;\
	ret; \
})
#define vec3Sub(v1, v2, t)({\
	t ret;\
	ret.x = v1.x - v2.x;\
	ret.y = v1.y - v2.y;\
	ret.z = v1.z - v2.z;\
	ret; \
})
#define vec3Mul(v1, v2, t)({\
	t ret;\
	ret.x = v1.x * v2.x;\
	ret.y = v1.y * v2.y;\
	ret.z = v1.z * v2.z;\
	ret; \
})
#define vec3Div(v1, v2, t)({\
	t ret;\
	ret.x = v1.x / v2.x;\
	ret.y = v1.y / v2.y;\
	ret.z = v1.z / v2.z;\
	ret; \
})


#define vec3AddC(v1, c, t)({\
	t ret;\
	ret.x = v1.x + c;\
	ret.y = v1.y + c;\
	ret.z = v1.z + c;\
	ret; \
})
#define vec3SubC(v1, c, t)({\
	t ret;\
	ret.x = v1.x - c;\
	ret.y = v1.y - c;\
	ret.z = v1.z - c;\
	ret; \
})
#define vec3MulC(v1, c, t)({\
	t ret;\
	ret.x = v1.x * c;\
	ret.y = v1.y * c;\
	ret.z = v1.z * c;\
	ret; \
})
#define vec3DivC(v1, c, t)({\
	t ret;\
	ret.x = v1.x / c;\
	ret.y = v1.y / c;\
	ret.z = v1.z / c;\
	ret; \
})


extern const Vec2i32 zero2i32;
extern const Vec3i32 zero3i32;
extern const Vec4i32 zero4i32;

extern const Vec2i32 one2i32;
extern const Vec3i32 one3i32;
extern const Vec4i32 one4i32;


extern const Vec2u32 zero2u32;
extern const Vec3u32 zero3u32;
extern const Vec4u32 zero4u32;

extern const Vec2u32 one2u32;
extern const Vec3u32 one3u32;
extern const Vec4u32 one4u32;


extern const Vec2u8 zero2u8;
extern const Vec3u8 zero3u8;
extern const Vec4u8 zero4u8;

extern const Vec2u8 one2i8;
extern const Vec3u8 one3i8;
extern const Vec4u8 one4i8;


extern const Vec2f zero2f;
extern const Vec3f zero3f;
extern const Vec4f zero4f;

extern const Vec2f one2f;
extern const Vec3f one3f;
extern const Vec4f one4f;