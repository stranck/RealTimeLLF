#pragma once

#include <math.h>

#define min(a, b) ({ (a < b) ? a : b; })
#define max(a, b) ({ (a > b) ? a : b; })

inline double clamp(double a, double min_, double max_) {
	double m = max(a, min_);
	return min(m, max_);
}

inline double smoothstep(double a, double b, double u) {
	double t = clamp((u - a) / (b - a), 0.0, 1.0);
	return t * t * (3 - 2 * t);
}