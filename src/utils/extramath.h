#pragma once

#include <math.h>

inline double min(double a, double b) { return (a < b) ? a : b; }
inline double max(double a, double b) { return (a > b) ? a : b; }

inline double clamp(double a, double min_, double max_) {
	return min(max(a, min_), max_);
}

inline double smoothstep(double a, double b, double u) {
	double t = clamp((u - a) / (b - a), 0.0, 1.0);
	return t * t * (3 - 2 * t);
}