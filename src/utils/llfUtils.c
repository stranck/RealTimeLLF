#include "llfUtils.h"

#include "../utils/imageutils.h"
#include "../utils/extramath.h"
#include "../utils/vects.h"
#include "../utils/utils.h"

#include <math.h>

void remap(Image3 * img, Pixel3 g0, double sigma, double alpha, double beta){
	int size = img -> width * img -> height;
	Pixel3 *pixels = img -> pixels;
	for(int i = 0; i < size; i++){
		Pixel3 delta = vec3Sub(pixels[i], g0, Pixel3);
		double mag = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
		if(mag > 1e-10) {
			delta = vec3DivC(delta, mag, Pixel3);
		}

		if(mag < sigma){ //Details
			double fraction = mag / sigma;
			double polynomial = pow(fraction, alpha);
			if(alpha < 1){
				const double kNoiseLevel = 0.01;
				double blend = smoothstep(kNoiseLevel, 2 * kNoiseLevel, fraction * sigma);
				polynomial = blend * polynomial + (1 - blend) * fraction;
			}
			double d = sigma * polynomial;
			Pixel3 px = vec3MulC(delta, d, Pixel3);
			img -> pixels[i] = vec3Add(g0, px, Pixel3);
		} else { //Edges
			double d = ((mag - sigma) * beta) + sigma;
			Pixel3 px = vec3MulC(delta, d, Pixel3);
			img -> pixels[i] = vec3Add(g0, px, Pixel3);
		}
	}
}

Kernel createFilter(){
	const int kernelDimension = 5;
	const double params[kernelDimension] = {0.05, 0.25, 0.4, 0.25, 0.05};
	Kernel filter = malloc(kernelDimension * kernelDimension * sizeof(double));

	for(uint8_t i = 0; i < 5; i++){
		for(uint8_t j = 0; j < 5; j++){
			filter[i][j] = params[i] * params[j];
		}
	}
	return filter;
}