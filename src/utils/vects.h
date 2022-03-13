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