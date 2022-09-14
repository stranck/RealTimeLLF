#pragma once

#include <math.h>

#define llf_min(a, b) ( (a < b) ? a : b )
#define llf_max(a, b) ( (a > b) ? a : b )

inline float clamp(float a, float min_, float max_) {
	float m = llf_max(a, min_);
	return llf_min(m, max_);
}

inline float smoothstep(float a, float b, float u) {
	float t = clamp((u - a) / (b - a), 0.0, 1.0);
	return t * t * (3 - 2 * t);
}