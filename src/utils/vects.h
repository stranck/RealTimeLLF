#include <stdint.h>

#ifndef VECTS_DEP
#define VECTS_DEP

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

const Vec2i32 zero2i32 = {0, 0};
const Vec3i32 zero3i32 = {0, 0, 0};
const Vec4i32 zero4i32 = {0, 0, 0, 0};

const Vec2i32 one2i32 = {1, 1};
const Vec3i32 one3i32 = {1, 1, 1};
const Vec4i32 one4i32 = {1, 1, 1, 1};

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

const Vec2u32 zero2u32 = {0, 0};
const Vec3u32 zero3u32 = {0, 0, 0};
const Vec4u32 zero4u32 = {0, 0, 0, 0};

const Vec2u32 one2u32 = {1, 1};
const Vec3u32 one3u32 = {1, 1, 1};
const Vec4u32 one4u32 = {1, 1, 1, 1};

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

const Vec2u8 zero2u8 = {0, 0};
const Vec3u8 zero3u8 = {0, 0, 0};
const Vec4u8 zero4u8 = {0, 0, 0, 0};

const Vec2u8 one2i8 = {1, 1};
const Vec3u8 one3i8 = {1, 1, 1};
const Vec4u8 one4i8 = {1, 1, 1, 1};

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

const Vec2f zero2f = {0, 0};
const Vec3f zero3f = {0, 0, 0};
const Vec4f zero4f = {0, 0, 0, 0};

const Vec2f one2f = {1, 1};
const Vec3f one3f = {1, 1, 1};
const Vec4f one4f = {1, 1, 1, 1};

#endif