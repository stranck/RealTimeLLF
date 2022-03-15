#include "llfUtils.h"

#include "../utils/imageutils.h"
#include "../utils/extramath.h"
#include "../utils/vects.h"
#include "../utils/utils.h"

#include <math.h>

void remap(Image3 * img, const Pixel3 g0, double sigma, double alpha, double beta){
	uint32_t size = img -> width * img -> height;
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
	const double params[KERNEL_DIMENSION] = {0.05, 0.25, 0.4, 0.25, 0.05};
	Kernel filter = malloc(KERNEL_DIMENSION * KERNEL_DIMENSION * sizeof(double));

	for(uint8_t i = 0; i < 5; i++){
		for(uint8_t j = 0; j < 5; j++){
			filter[i][j] = params[i] * params[j];
		}
	}
	return filter;
}

Pyramid createPyramid(uint32_t width, uint32_t height, uint8_t nLevels){ //Pyramids has one more layer!
	Pyramid p = malloc(nLevels * sizeof(Image3*));
	for(uint8_t i = 0; i <= nLevels; i++){
		p[i] = makeImage3(width, height);
		width = width / 2 + (width & 1);
		height = height / 2 + (height & 1);
	}
	return p;
}

void convolve(Image3 *dest, Image3 *source, Kernel kernel) {
	uint8_t  rows = KERNEL_DIMENSION;
	uint8_t  cols = KERNEL_DIMENSION;
	int32_t  xstart = -1 * cols / 2;
	int32_t  ystart = -1 * rows / 2;

	for (uint32_t j = 0; j < dest->height; j++) {
		for (uint32_t i = 0; i < dest->width; i++) {
			Pixel3 c = zero3f;
			for (uint32_t y = 0; y < rows; y++) {
                int32_t jy = j + ystart + y;
				for (uint32_t x = 0; x < cols; x++) {
                    int32_t ix = i + xstart + x;
                    if (ix >= 0 && ix < dest->width && jy >= 0 && jy < dest->height) {
						double kern_elem = kernel[x][y];
						Pixel3 px = *getPixel3(source, ix, jy);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					} else {
						double kern_elem = kernel[x][y];
						Pixel3 px = *getPixel3(source, i, j);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					}
				}
			}
			setPixel3(dest, i, j, &c);
		}
	}
}