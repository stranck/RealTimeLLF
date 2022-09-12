#include "cudaUtils.cuh"

__host__ Kernel createFilterDevice(){
	const float params[KERNEL_DIMENSION] = {0.05, 0.25, 0.4, 0.25, 0.05};
	float h_filter[KERNEL_DIMENSION * KERNEL_DIMENSION];

	for(uint8_t i = 0; i < KERNEL_DIMENSION; i++){
		for(uint8_t j = 0; j < KERNEL_DIMENSION; j++){
			h_filter[getKernelPosition(i, j)] = params[i] * params[j];
		}
	}

	Kernel d_filter;
	CHECK(cudaMalloc((void**) &d_filter, KERNEL_DIMENSION * KERNEL_DIMENSION * sizeof(float)));
	printff("D_FILTER ADDR: 0x%016llx\n", d_filter);
	CHECK(cudaMemcpy(d_filter, h_filter, KERNEL_DIMENSION * KERNEL_DIMENSION * sizeof(float), cudaMemcpyHostToDevice));
	return d_filter;
}
__host__ void destroyFilterDevice(Kernel d_k){
	CHECK(cudaFree(d_k));
}

__device__ Pyramid d_createPyramid(uint32_t width, uint32_t height, uint8_t nLevels){
	nLevels++; //Pyramids has one more layer!
	Pyramid p;
	cudaMalloc(&p, nLevels * sizeof(Image3*));
	printf("d_createPyramid: Dimensions: %03dx%03d @ %d levels    Pyramid at 0x%012llx\n", width, height, nLevels, p);
	for(uint8_t i = 0; i < nLevels; i++){
		p[i] = d_makeImage3(width, height);
		width = width / 2 + (width & 1);
		height = height / 2 + (height & 1);
	}
	return p;
}
__host__ Pyramid createPyramidDevice(uint32_t width, uint32_t height, uint8_t nLevels){
	nLevels++; //Pyramids has one more layer!
	Pyramid h_p = (Pyramid) allocStack(nLevels * sizeof(Image3*));
	for(uint8_t i = 0; i < nLevels; i++){
		h_p[i] = makeImage3Device(width, height);
		width = width / 2 + (width & 1);
		height = height / 2 + (height & 1);
	}

	Pyramid d_p;
	CHECK(cudaMalloc((void**) &d_p, nLevels * sizeof(Image3*)));
	printff("CreatePyramidDevice: malloc pyramid at 0x%032llx. Params: %u\n", d_p, nLevels);
	CHECK(cudaMemcpy(d_p, h_p, nLevels * sizeof(Image3*), cudaMemcpyHostToDevice));
	return d_p;
}
__device__ void d_destroydPyramid(Pyramid pyr, uint8_t nLevels){
	for(uint8_t i = 0; i <= nLevels; i++)
		d_destroyImage3(pyr[i]);
	cudaFree(pyr);
}
__host__ void destroyPyramidDevice(Pyramid d_pyr, uint8_t h_nLevels){
	Pyramid h_pyr = (Pyramid) allocStack((h_nLevels + 1)* sizeof(Image3*));
	CHECK(cudaMemcpy(h_pyr, d_pyr, (h_nLevels + 1) * sizeof(Image3*), cudaMemcpyDeviceToHost));
	for(uint8_t i = 0; i <= h_nLevels; i++)
		destroyImage3Device(h_pyr[i]);
	CHECK(cudaFree(d_pyr));
}

__device__ Image3 * d_makeImage3(uint32_t width, uint32_t height){
	if(blockIdx.x == 0 && threadIdx.x == 0){
		Image3 *i;
		Pixel3 *img;
		cudaError_t errImg = cudaMalloc(&i, sizeof(Image3));
		cudaError_t errPx = cudaMalloc(&img, width * height * sizeof(Pixel3));
		i -> width = width;
		i -> height = height;
		i -> pixels = img;
		printf("d_makeImage3: Dimensions: % 3dx% 3d    Pixels at 0x%012llx    Image3 at 0x%012llx    Error img: %s     Error pxs: %s\n", width, height, i, img, cudaGetErrorString(errImg), cudaGetErrorString(errPx));
		return i;
	}
}
__host__ Image3 * makeImage3Device(uint32_t width, uint32_t height){
	Pixel3 *d_img;
	CHECK(cudaMalloc((void**) &d_img, width * height * sizeof(Pixel3)));
	Image3 h_i;
	h_i.width = width;
	h_i.height = height;
	h_i.pixels = d_img;

	Image3 *d_i;
	CHECK(cudaMalloc((void**) &d_i, sizeof(Image3)));
	CHECK(cudaMemcpy(d_i, &h_i, sizeof(Image3), cudaMemcpyHostToDevice));
	return d_i;
}
__device__ void d_destroyImage3(Image3 *img){
	cudaFree(img -> pixels);
	cudaFree(img);
}
__host__ void destroyImage3Device(Image3 *d_img){
	Image3 h_img;
	CHECK(cudaMemcpy(&h_img, d_img, sizeof(Image3), cudaMemcpyDeviceToHost));
	CHECK(cudaFree(h_img.pixels));
	CHECK(cudaFree(d_img));
}

__host__ void copyImg3Host2Device(Image3 *d_imgDst, Image3 *h_imgSrc){
	Image3 h_i;
	CHECK(cudaMemcpy(&h_i, d_imgDst, sizeof(Image3), cudaMemcpyDeviceToHost));
	h_i.width = h_imgSrc->width;
	h_i.height = h_imgSrc->height;
	CHECK(cudaMemcpy(d_imgDst, &h_i, sizeof(Image3), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(h_i.pixels, h_imgSrc->pixels, h_imgSrc->width * h_imgSrc->height * sizeof(Pixel3), cudaMemcpyHostToDevice));
}
__host__ void copyImg3Device2Host(Image3 *h_imgDst, Image3 *d_imgSrc){
	Image3 h_i;
	CHECK(cudaMemcpy(&h_i, d_imgSrc, sizeof(Image3), cudaMemcpyDeviceToHost));
	h_imgDst->width = h_i.width;
	h_imgDst->height = h_i.height;
	size_t h_toCopy = (h_i.width) * (h_i.height) * sizeof(Pixel3);
	CHECK(cudaMemcpy(h_imgDst->pixels, h_i.pixels, h_toCopy, cudaMemcpyDeviceToHost));
}
__device__ void d_imgcpy3(Image3 *d_dest, Image3 *d_source){
	__shared__ uint32_t dim;
	__shared__ Pixel3* d_destPxs;
	__shared__ Pixel3* d_srcPxs;

	if(threadIdx.x == 0){
		d_dest->width = d_source->width;
		d_dest->height = d_source->height;
		dim = d_dest->width * d_dest->height;
		d_destPxs = d_dest->pixels;
		d_srcPxs = d_source->pixels;
	}
	__syncthreads();
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim)
			d_destPxs[idx] = d_srcPxs[idx];
	}
	__syncthreads();
}
__global__ void d_copyPyrLevel(Pyramid dst_pyr, Pyramid src_pyr, uint8_t level){
	d_imgcpy3(dst_pyr[level], src_pyr[level]);
} 

__host__ Image3 * getImageFromPyramidDevice(Pyramid d_pyr, uint8_t h_level){
	Pyramid h_pyr = (Pyramid) allocStack((h_level + 1) * sizeof(Image3*)); //We just need to copy up to level pointers;
	CHECK(cudaMemcpy(h_pyr, d_pyr, (h_level + 1) * sizeof(Image3*), cudaMemcpyDeviceToHost));
	return h_pyr[h_level];
}
__host__ void getPyramidDimensionsAtLayer(Pyramid d_pyr, uint8_t h_level, uint32_t *h_width, uint32_t *h_height){
	Image3 h_lvl;
	Image3 *d_img = getImageFromPyramidDevice(d_pyr, h_level);
	CHECK(cudaMemcpy(&h_lvl, d_img, sizeof(Image3), cudaMemcpyDeviceToHost));
	*h_width = h_lvl.width;
	*h_height = h_lvl.height;
}

__device__ void d_subimage3Remap_shared(Pixel3 *destPx, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY, const Pixel3 g0, float sigma, float alpha, float beta){
	uint32_t w = endX - startX;
	uint32_t h = endY - startY;

	Pixel3 *srcPx = source->pixels;
	uint32_t srcW = source->width;
	uint32_t dim = w * h;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t x = idx % w, y = idx / w;
			uint32_t finalY = startY + y;

			Pixel3 p = d_getPixel3(srcPx, srcW, startX + x, finalY);
			Pixel3 remapped = d_remapSinglePixel(p, g0, sigma, alpha, beta);
			d_setPixel3(destPx, w, x, y, remapped);
		}
	}
	__syncthreads();
}
__device__ void d_subimage3Remap(Image3 *dest, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY, const Pixel3 g0, float sigma, float alpha, float beta){
	uint32_t w = endX - startX;
	uint32_t h = endY - startY;
	if(threadIdx.x == 0){
		dest->width = w;
		dest->height = h;
	}

	Pixel3 *destPx = dest->pixels, *srcPx = source->pixels;
	uint32_t srcW = source->width;
	uint32_t dim = w * h;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t x = idx % w, y = idx / w;
			uint32_t finalY = startY + y;

			Pixel3 p = d_getPixel3(srcPx, srcW, startX + x, finalY);
			Pixel3 remapped = d_remapSinglePixel(p, g0, sigma, alpha, beta);
			d_setPixel3(destPx, w, x, y, remapped);
		}
	}
	__syncthreads();
}
__device__ void d_subimage3(Image3 *dest, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY){
	uint32_t w = endX - startX;
	uint32_t h = endY - startY;
	if(threadIdx.x == 0){
		dest->width = w;
		dest->height = h;
	}

	Pixel3 *destPx = dest->pixels, *srcPx = source->pixels;
	uint32_t srcW = source->width;
	uint32_t dim = w * h;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t x = idx % w, y = idx / w;
			uint32_t finalY = startY + y;
			d_setPixel3(destPx, w, x, y, d_getPixel3(srcPx, srcW, startX + x, finalY));
		}
	}
	__syncthreads();
}

__global__ void d_clampImage3(Image3 *img){
	__shared__ uint32_t dim;
	__shared__ Pixel3 *px;

	if(threadIdx.x == 0){
		dim = img->width * img->height;
		px = img->pixels;
	}
	__syncthreads();

	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < dim){
		px[i].x = d_clamp(px[i].x, 0, 1);
		px[i].y = d_clamp(px[i].y, 0, 1);
		px[i].z = d_clamp(px[i].z, 0, 1);
	}
	__syncthreads();
}
__device__ float d_clamp(float a, float min_, float max_) {
	int minFlag = a < min_;
	int maxFlag = a > max_;
	int flag = minFlag + maxFlag;
	//if(flag > 1) flag = 1; //no way they are both true at the same time IF THE PARAMS ARE CORRECT :<
	return a * (1 - flag) + min_ * minFlag + max_ * maxFlag;
}
__device__ float d_smoothstep(float a, float b, float u) {
	float t = d_clamp((u - a) / (b - a), 0.0, 1.0);
	return t * t * (3 - 2 * t);
}

__device__ inline Pixel3 d_remapSinglePixel(const Pixel3 source, const Pixel3 g0, float sigma, float alpha, float beta){
	Pixel3 delta;
	vec3Sub(delta, source, g0);
	float mag = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
	if(mag > 1e-10) vec3DivC(delta, delta, mag);

	int details = mag < sigma;
	float fraction = mag / sigma;
	float polynomial = pow(fraction, alpha);
	if(alpha < 1){ //alpha is one of the entire llf params, so ALL the threads will always take the same branch
		const float kNoiseLevel = 0.01;
		float blend = d_smoothstep(kNoiseLevel, 2 * kNoiseLevel, fraction * sigma);
		polynomial = blend * polynomial + (1 - blend) * fraction;
	}
	float d = (sigma * polynomial) * details + (((mag - sigma) * beta) + sigma) * (1 - details);
	vec3MulC(delta, delta, d);
	vec3Add(delta, g0, delta);
	return delta;
}
__device__ void d_remap(Image3 * img, const Pixel3 g0, float sigma, float alpha, float beta){
	uint32_t dim = img -> width * img -> height;
	uint32_t max = dim / blockDim.x;
	Pixel3 *pixels = img -> pixels;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim)
			pixels[idx] = d_remapSinglePixel(pixels[idx], g0, sigma, alpha, beta);
	}
	__syncthreads();
}