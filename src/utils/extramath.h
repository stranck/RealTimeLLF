#pragma once

/**
 * @brief returns the min between two numbers
 */
#define llf_min(a, b) ( (a < b) ? a : b )
/**
 * @brief returns the max between two numbers
 */
#define llf_max(a, b) ( (a > b) ? a : b )

/**
 * @brief forces the input to be inside of the specified range
 * 
 * @param a input
 * @param min_ min value of the range
 * @param max_ max value of the range
 * @return float a if it's between min and max, min if a < min, max if a > max
 */
inline float clamp(float a, float min_, float max_) {
	float m = llf_max(a, min_);
	return llf_min(m, max_);
}

/**
 * @brief Applies a smoothstep function
 */
inline float smoothstep(float a, float b, float u) {
	float t = clamp((u - a) / (b - a), 0.0, 1.0);
	return t * t * (3 - 2 * t);
}