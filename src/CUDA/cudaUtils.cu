#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudaStructs.cuh"
#include "cudaUtils.cuh"
#include "../utils/utils.h"
#include "../utils/llfUtils.h"
#include "../utils/structs.h"
#include "../utils/imageutils.h"
#include "../utils/extramath.h"


Kernel createFilterDevice(){
	const double params[KERNEL_DIMENSION] = {0.05, 0.25, 0.4, 0.25, 0.05};
	Kernel h_filter = (Kernel) malloc(KERNEL_DIMENSION * KERNEL_DIMENSION * sizeof(double));

	for(uint8_t i = 0; i < KERNEL_DIMENSION; i++){
		for(uint8_t j = 0; j < KERNEL_DIMENSION; j++){
			h_filter[getKernelPosition(i, j)] = params[i] * params[j];
		}
	}

	Kernel d_filter;
	CHECK(cudaMalloc((void**) &d_filter, KERNEL_DIMENSION * KERNEL_DIMENSION * sizeof(double)));
	CHECK(cudaMemcpy(d_filter, h_filter, KERNEL_DIMENSION * KERNEL_DIMENSION * sizeof(double), cudaMemcpyHostToDevice));
	free(h_filter);
	return d_filter;
}

Pyramid createPyramidDevice(uint32_t width, uint32_t height, uint8_t nLevels){
	nLevels++; //Pyramids has one more layer!
	Pyramid h_p = (Pyramid) malloc(nLevels * sizeof(Image3*));
	for(uint8_t i = 0; i < nLevels; i++){
		h_p[i] = makeImage3Device(width, height);
		width = width / 2 + (width & 1);
		height = height / 2 + (height & 1);
	}

	Pyramid d_p;
	CHECK(cudaMalloc((void**) &d_p, nLevels * sizeof(Image3*)));
	CHECK(cudaMemcpy(d_p, h_p, nLevels * sizeof(Image3*), cudaMemcpyHostToDevice));
	free(h_p);
	return d_p;
}

Image3 * makeImage3Device(uint32_t width, uint32_t height){
	Pixel3 *d_img;
	CHECK(cudaMalloc((void**) &d_img, width * height * sizeof(Pixel3)));
	Image3 *h_i = (Image3 *) malloc(sizeof(Image3));
	h_i -> width = width;
	h_i -> height = height;
	h_i -> pixels = d_img;

	Image3 *d_i;
	CHECK(cudaMalloc((void**) &d_i, sizeof(Image3)));
	CHECK(cudaMemcpy(d_i, h_i, sizeof(Image3), cudaMemcpyHostToDevice));
	free(h_i);
	return d_i;
}
Image3 * copyImg3Host2Device(Image3 * h_img){
	Pixel3 *d_img;
	CHECK(cudaMalloc((void**) &d_img, h_img->width * h_img->height * sizeof(Pixel3)));
	CHECK(cudaMemcpy(d_img, h_img->pixels, h_img->width * h_img->height * sizeof(Pixel3), cudaMemcpyHostToDevice));
	Image3 *h_i = (Image3 *) malloc(sizeof(Image3));
	h_i -> width = h_img->width;
	h_i -> height = h_img->height;
	h_i -> pixels = d_img;

	Image3 *d_i;
	CHECK(cudaMalloc((void**) &d_i, sizeof(Image3)));
	CHECK(cudaMemcpy(d_i, h_i, sizeof(Image3), cudaMemcpyHostToDevice));
	free(h_i);
	return d_i;
}

//__device__ 

__device__ void d_imgcpy3(Image3 *d_dest, Image3 *d_source){
	if(threadIdx.x == 0){
		d_dest->width = d_source->width;
		d_dest->height = d_source->height;
		//CHECK(cudaMemcpy(d_dest->pixels, d_source->pixels, d_dest->width * d_dest->height * sizeof(Pixel3), cudaMemcpyDeviceToDevice));
	}
	__syncthreads();
	uint32_t dim = d_dest->width * d_dest->height;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim)
			d_dest->pixels[idx] = d_source->pixels[idx];
	}
	__syncthreads();
}

__device__ void d_subimage3(Image3 *dest, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY){
	uint32_t w = endX - startX;
	uint32_t h = endY - startY;
	dest->width = w;
	dest->height = h;

	uint32_t dim = w * h;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t x = idx % w, y = idx / w;
	//for(uint32_t y = 0; y < h; y++){
			uint32_t finalY = startY + y;
		//for(uint32_t x = 0; x < w; x++){
			setPixel3(dest, x, y, getPixel3(source, startX + x, finalY));
		}
	}
}

__device__ double d_clamp(double a, double min_, double max_) {
	int minFlag = a < min_;
	int maxFlag = a > max_;
	int flag = minFlag + maxFlag;
	//if(flag > 1) flag = 1; //no way they are both true at the same time IF THE PARAMS ARE CORRECT :<
	return a * (1 - flag) + min_ * minFlag + max_ * maxFlag;
}
__device__ double d_smoothstep(double a, double b, double u) {
	double t = d_clamp((u - a) / (b - a), 0.0, 1.0);
	return t * t * (3 - 2 * t);
}

__device__ void d_remap(Image3 * img, const Pixel3 g0, double sigma, double alpha, double beta){
	uint32_t dim = img -> width * img -> height;
	uint32_t max = dim / blockDim.x;
	Pixel3 *pixels = img -> pixels;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim){

			Pixel3 delta = vec3Sub(pixels[idx], g0, Pixel3);
			double mag = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
			if(mag > 1e-10) delta = vec3DivC(delta, mag, Pixel3);

			int details = mag < sigma;
			double fraction = mag / sigma;
			double polynomial = pow(fraction, alpha);
			if(alpha < 1){ //alpha is one of the entire llf params, so ALL the threads will always take the same branch
				const double kNoiseLevel = 0.01;
				double blend = d_smoothstep(kNoiseLevel, 2 * kNoiseLevel, fraction * sigma);
				polynomial = blend * polynomial + (1 - blend) * fraction;
			}
			double d = (sigma * polynomial) * details + (((mag - sigma) * beta) + sigma) * (1 - details);
			Pixel3 px = vec3MulC(delta, d, Pixel3);
			pixels[idx] = vec3Add(g0, px, Pixel3);

			/*if(mag < sigma){ //Details
				double fraction = mag / sigma;
				double polynomial = pow(fraction, alpha);
				if(alpha < 1){ //alpha is one of the entire llf params, so ALL the threads will always take the same branch
					const double kNoiseLevel = 0.01;
					double blend = d_smoothstep(kNoiseLevel, 2 * kNoiseLevel, fraction * sigma);
					polynomial = blend * polynomial + (1 - blend) * fraction;
				}
				double d = sigma * polynomial;
				Pixel3 px = vec3MulC(delta, d, Pixel3);
				img -> pixels[idx] = vec3Add(g0, px, Pixel3);
			} else { //Edges
				double d = ((mag - sigma) * beta) + sigma;
				Pixel3 px = vec3MulC(delta, d, Pixel3);
				img -> pixels[idx] = vec3Add(g0, px, Pixel3);
			}*/
		}
	}
}