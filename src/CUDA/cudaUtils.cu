#include "cudaUtils.cuh"

/**
 * @brief Allocates and computes the blur kernel on the device
 * 
 * @return Kernel device-allocated blur kernel 
 */
__host__ Kernel createFilterDevice(){
	const float params[KERNEL_DIMENSION] = {0.05, 0.25, 0.4, 0.25, 0.05};
	float h_filter[KERNEL_DIMENSION * KERNEL_DIMENSION]; //alloc the kernel on the host's stack

	//compute the kernel
	for(uint8_t i = 0; i < KERNEL_DIMENSION; i++){
		for(uint8_t j = 0; j < KERNEL_DIMENSION; j++){
			h_filter[getKernelPosition(i, j)] = params[i] * params[j];
		}
	}

	Kernel d_filter;
	CHECK(cudaMalloc((void**) &d_filter, KERNEL_DIMENSION * KERNEL_DIMENSION * sizeof(float))); //alloc the kernel on the device
	printff("D_FILTER ADDR: 0x%016llx\n", d_filter);
	CHECK(cudaMemcpy(d_filter, h_filter, KERNEL_DIMENSION * KERNEL_DIMENSION * sizeof(float), cudaMemcpyHostToDevice)); //Copy the kernel to the device
	return d_filter;
}
/**
 * @brief frees a kernel from the device memory
 * 
 * @param d_k device-allocated kernel
 */
__host__ void destroyFilterDevice(Kernel d_k){
	CHECK(cudaFree(d_k));
}

/**
 * @brief Allocates a pyramid on the device's memory. Must be called from the device
 * 
 * @param width width of the layer 0
 * @param height height of the layer 0
 * @param nLevels number of levels in the pyramid
 * @return Pyramid newly allocated pyramid in the device's memory
 */
__device__ Pyramid d_createPyramid(uint32_t width, uint32_t height, uint8_t nLevels){
	nLevels++; //Pyramids has one more layer!
	Pyramid p;
	cudaMalloc(&p, nLevels * sizeof(Image3*));
	printf("d_createPyramid: Dimensions: %03dx%03d @ %d levels    Pyramid at 0x%012llx\n", width, height, nLevels, p);
	for(uint8_t i = 0; i < nLevels; i++){
		p[i] = d_makeImage3(width, height);
		width = width / 2 + (width & 1); //Halfs the size of each layer
		height = height / 2 + (height & 1);
	}
	return p;
}
/**
 * @brief Allocates a pyramid on the device's memory. Must be called from the host
 * 
 * @param width width of the layer 0
 * @param height height of the layer 0
 * @param nLevels number of levels in the pyramid
 * @return Pyramid newly allocated pyramid in the device's memory
 */
__host__ Pyramid createPyramidDevice(uint32_t width, uint32_t height, uint8_t nLevels){
	nLevels++; //Pyramids has one more layer!
	Pyramid h_p = (Pyramid) allocStack(nLevels * sizeof(Image3*)); //Allocates the pyramid in the host's stack
	for(uint8_t i = 0; i < nLevels; i++){
		h_p[i] = makeImage3Device(width, height); //Allocate the single images on the device memory and copy their ptr on the host's stack
		width = width / 2 + (width & 1); //Halfs the size of each layer
		height = height / 2 + (height & 1);
	}

	Pyramid d_p;
	CHECK(cudaMalloc((void**) &d_p, nLevels * sizeof(Image3*))); //Allocates the pyramid in the device's memory
	printff("CreatePyramidDevice: malloc pyramid at 0x%032llx. Params: %u\n", d_p, nLevels);
	CHECK(cudaMemcpy(d_p, h_p, nLevels * sizeof(Image3*), cudaMemcpyHostToDevice)); //Copy the pyramid containing pointers to images on the device, to the device's memory
	return d_p;
}
/**
 * @brief Frees a pyramid from the device's memory. Must be called from the device
 * 
 * @param pyr Pyramid to be freed 
 * @param nLevels Number of levels of the pyramid
 */
__device__ void d_destroydPyramid(Pyramid pyr, uint8_t nLevels){
	for(uint8_t i = 0; i <= nLevels; i++)
		d_destroyImage3(pyr[i]);
	cudaFree(pyr);
}
/**
 * @brief Frees a pyramid from the device's memory. Must be called from the host
 * 
 * @param d_pyr Pyramid to be freed
 * @param h_nLevels Number of levels of the pyramid
 */
__host__ void destroyPyramidDevice(Pyramid d_pyr, uint8_t h_nLevels){
	Pyramid h_pyr = (Pyramid) allocStack((h_nLevels + 1)* sizeof(Image3*)); //Allocates the pyramid on the host's stack
	CHECK(cudaMemcpy(h_pyr, d_pyr, (h_nLevels + 1) * sizeof(Image3*), cudaMemcpyDeviceToHost)); //Copy the pyramid to the host's memory
	for(uint8_t i = 0; i <= h_nLevels; i++)
		destroyImage3Device(h_pyr[i]); //Destroys each single image on the device
	CHECK(cudaFree(d_pyr)); //Frees the pyramid
}

/**
 * @brief Allocates an image3 on the device's memory. Must be called from the device
 * 
 * @param width width of the image
 * @param height height of the image
 * @return Image3* Newly allocated image3 on the device's memory
 */
__device__ Image3 * d_makeImage3(uint32_t width, uint32_t height){
	Image3 *i = NULL;
	if(blockIdx.x == 0 && threadIdx.x == 0){ //Only one thread in the kernel must execute that call!
		Pixel3 *img;
		cudaError_t errImg = cudaMalloc(&i, sizeof(Image3));
		cudaError_t errPx = cudaMalloc(&img, width * height * sizeof(Pixel3));
		i -> width = width;
		i -> height = height;
		i -> pixels = img;
		printf("d_makeImage3: Dimensions: % 3dx% 3d    Pixels at 0x%012llx    Image3 at 0x%012llx    Error img: %s     Error pxs: %s\n", width, height, i, img, cudaGetErrorString(errImg), cudaGetErrorString(errPx));
	}
	return i;
}
/**
 * @brief Allocates an image3 on the device's memory. Must be called from the host
 * 
 * @param width width of the image
 * @param height height of the image
 * @return Image3* Newly allocated image3 on the device's memory
 */
__host__ Image3 * makeImage3Device(uint32_t width, uint32_t height){
	Pixel3 *d_img;
	CHECK(cudaMalloc((void**) &d_img, width * height * sizeof(Pixel3))); //Allocates the pixel's buffer in the device memory
	Image3 h_i; //Creates the image first in the host's stack
	h_i.width = width;
	h_i.height = height;
	h_i.pixels = d_img; //Set the pointer to the pixel buffer inside the device's memory

	Image3 *d_i;
	CHECK(cudaMalloc((void**) &d_i, sizeof(Image3))); //Allocates the images on the device's memory
	CHECK(cudaMemcpy(d_i, &h_i, sizeof(Image3), cudaMemcpyHostToDevice)); //Copies the whole image struct to the device
	return d_i;
}
/**
 * @brief Frees an image from the device's memory. Must be called from the device
 * 
 * @param img image to free
 */
__device__ void d_destroyImage3(Image3 *img){
	cudaFree(img -> pixels);
	cudaFree(img);
}
/**
 * @brief Frees an image from the device's memory. Must be called from the host
 * 
 * @param img image to free
 */
__host__ void destroyImage3Device(Image3 *d_img){
	Image3 h_img;
	CHECK(cudaMemcpy(&h_img, d_img, sizeof(Image3), cudaMemcpyDeviceToHost)); //Copy the image to the host's stack, to get the pointer to the pixel buffer
	CHECK(cudaFree(h_img.pixels));
	CHECK(cudaFree(d_img));
}
/**
 * @brief Copies an image from the host's memory to the device's memory
 * 
 * @param d_imgDst Already allocated image on the device's memory
 * @param h_imgSrc Source image on the host's memory
 */
__host__ void copyImg3Host2Device(Image3 *d_imgDst, Image3 *h_imgSrc){
	Image3 h_i;
	CHECK(cudaMemcpy(&h_i, d_imgDst, sizeof(Image3), cudaMemcpyDeviceToHost)); //Copies the device's image to the host's memory to get the address of the pixel buffer
	h_i.width = h_imgSrc->width;
	h_i.height = h_imgSrc->height;
	CHECK(cudaMemcpy(d_imgDst, &h_i, sizeof(Image3), cudaMemcpyHostToDevice)); //Copies the updated dimensions to the device's memory
	CHECK(cudaMemcpy(h_i.pixels, h_imgSrc->pixels, h_imgSrc->width * h_imgSrc->height * sizeof(Pixel3), cudaMemcpyHostToDevice)); //copies the pixel buffer to the device's memory
}
/**
 * @brief Copies an image from the device's memory to the host's memory
 * 
 * @param d_imgDst Already allocated image on the device's memory
 * @param h_imgSrc Source image on the host's memory
 */
__host__ void copyImg3Device2Host(Image3 *h_imgDst, Image3 *d_imgSrc){
	Image3 h_i;
	CHECK(cudaMemcpy(&h_i, d_imgSrc, sizeof(Image3), cudaMemcpyDeviceToHost)); //Copies the whole device image to a temp one in the host's stack, to get its dimension the the address of the pixel buffer
	h_imgDst->width = h_i.width;
	h_imgDst->height = h_i.height;
	size_t h_toCopy = (h_i.width) * (h_i.height) * sizeof(Pixel3);
	CHECK(cudaMemcpy(h_imgDst->pixels, h_i.pixels, h_toCopy, cudaMemcpyDeviceToHost)); //Copies the pixel buffer to the host's memory
}
/**
 * @brief Copies an image from device to device
 * 
 * @param d_dest Dest image
 * @param d_source Source image
 */
__device__ void d_imgcpy3(Image3 *d_dest, Image3 *d_source){
	__shared__ uint32_t dim;
	__shared__ Pixel3* d_destPxs;
	__shared__ Pixel3* d_srcPxs;

	if(threadIdx.x == 0){ //Use only one thread to access global memory to copy the dimensions and get the addresses of the pixel buffer
		d_dest->width = d_source->width;
		d_dest->height = d_source->height;
		dim = d_dest->width * d_dest->height;
		d_destPxs = d_dest->pixels;
		d_srcPxs = d_source->pixels;
	}
	__syncthreads();
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){ //Use multiple threads in the same block to copy the pixels. Since we use this function inside the llf algorithm where each block works on a different pixel, we don't parallelize also for blocks
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim)
			d_destPxs[idx] = d_srcPxs[idx];
	}
	__syncthreads();
}
/**
 * @brief Copy one layer of a pyramid on the device's memory to another one at the same level
 * 
 * @param dst_pyr Dest pyramid
 * @param src_pyr Source pyramid
 * @param level Level to copy
 */
__global__ void d_copyPyrLevel(Pyramid dst_pyr, Pyramid src_pyr, uint8_t level){
	d_imgcpy3(dst_pyr[level], src_pyr[level]);
} 

/**
 * @brief Gets the pointer to a device allocated image from a layer of a device allocated pyramid
 * 
 * @param d_pyr Source pyramid
 * @param h_level Level on the pyramid
 * @return Image3* Pointer to the device allocated image at the specified layer of the pyramid
 */
__host__ Image3 * getImageFromPyramidDevice(Pyramid d_pyr, uint8_t h_level){
	Pyramid h_pyr = (Pyramid) allocStack((h_level + 1) * sizeof(Image3*)); //We just need to copy up to level pointers;
	CHECK(cudaMemcpy(h_pyr, d_pyr, (h_level + 1) * sizeof(Image3*), cudaMemcpyDeviceToHost)); //copy the whole pyramid to the host's memory
	return h_pyr[h_level];
}
/**
 * @brief Get the dimensions of a pyramid's layer
 * 
 * @param d_pyr source pyramid
 * @param h_level layer no
 * @param h_width output width
 * @param h_height output height
 */
__host__ void getPyramidDimensionsAtLayer(Pyramid d_pyr, uint8_t h_level, uint32_t *h_width, uint32_t *h_height){
	Image3 h_lvl;
	Image3 *d_img = getImageFromPyramidDevice(d_pyr, h_level); //Copy the interested layer to the host's memory
	CHECK(cudaMemcpy(&h_lvl, d_img, sizeof(Image3), cudaMemcpyDeviceToHost)); //Copy the image metadata to the host's stack
	*h_width = h_lvl.width;
	*h_height = h_lvl.height;
}

/**
 * @brief cuts a subimage from an image on the device's memory. It also directly applies the ramap function to save memory accesses
 * The destination is a pixel buffer that can also be in shared memory
 * 
 * @param destPx Destination pixel buffer. It should be in shared memory
 * @param source Source image
 * @param startX x base value of the subregion in the source image
 * @param endX x end value of the subregion in the source image
 * @param startY y base value of the subregion in the source image
 * @param endY y end value of the subregion in the source image
 * @param g0 Reference pixel
 * @param sigma Treshold used by remap function to identify edges and details
 * @param alpha Controls the details level
 * @param beta Controls the tone mapping level
 */
__device__ void d_subimage3Remap_shared(Pixel3 *destPx, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY, const Pixel3 g0, float sigma, float alpha, float beta){
	uint32_t w = endX - startX;
	uint32_t h = endY - startY;

	Pixel3 *srcPx = source->pixels;
	uint32_t srcW = source->width;
	uint32_t dim = w * h;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){ //Use multiple threads in the same block to copy the pixels. Since we use this function inside the llf algorithm where each block works on a different pixel, we don't parallelize also for blocks
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t x = idx % w, y = idx / w;
			uint32_t finalY = startY + y;

			Pixel3 p = d_getPixel3(srcPx, srcW, startX + x, finalY); //Take the pixel
			Pixel3 remapped = d_remapSinglePixel(p, g0, sigma, alpha, beta); //remap it
			d_setPixel3(destPx, w, x, y, remapped); //save it to the dest image
		}
	}
	__syncthreads();
}
/**
 * @brief cuts a subimage from an image on the device's memory. It also directly applies the ramap function to save memory accesses
 * 
 * @param destPx Destination image
 * @param source Source image
 * @param startX x base value of the subregion in the source image
 * @param endX x end value of the subregion in the source image
 * @param startY y base value of the subregion in the source image
 * @param endY y end value of the subregion in the source image
 * @param g0 Reference pixel
 * @param sigma Treshold used by remap function to identify edges and details
 * @param alpha Controls the details level
 * @param beta Controls the tone mapping level
 */
__device__ void d_subimage3Remap(Image3 *dest, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY, const Pixel3 g0, float sigma, float alpha, float beta){
	d_subimage3Remap_shared(dest->pixels, source, startX, endX, startY, endY, g0, sigma, alpha, beta);
}
/**
 * @brief cuts a subimage from an image on the device's memory
 * 
 * @param destPx Destination image
 * @param source Source image
 * @param startX x base value of the subregion in the source image
 * @param endX x end value of the subregion in the source image
 * @param startY y base value of the subregion in the source image
 * @param endY y end value of the subregion in the source image
 */
__device__ void d_subimage3(Image3 *dest, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY){
	uint32_t w = endX - startX;
	uint32_t h = endY - startY;
	if(threadIdx.x == 0){ //only one thread should update the dest image's dimension
		dest->width = w;
		dest->height = h;
	}

	Pixel3 *destPx = dest->pixels, *srcPx = source->pixels;
	uint32_t srcW = source->width;
	uint32_t dim = w * h;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){ //Use multiple threads in the same block to copy the pixels. Since we use this function inside the llf algorithm where each block works on a different pixel, we don't parallelize also for blocks
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t x = idx % w, y = idx / w;
			uint32_t finalY = startY + y;
			d_setPixel3(destPx, w, x, y, d_getPixel3(srcPx, srcW, startX + x, finalY));
		}
	}
	__syncthreads();
}

/**
 * @brief Clamps an image3 on the gpu to have each single pixel in the [0;1] boundaries
 * 
 * @param img Image to clamp on the device's memory
 */
__global__ void d_clampImage3(Image3 *img){
	__shared__ uint32_t dim;
	__shared__ Pixel3 *px;

	if(threadIdx.x == 0){ //only one thread per block will load the image metadata
		dim = img->width * img->height;
		px = img->pixels;
	}
	__syncthreads();

	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < dim){ //each thread of every block will clamp each pixel
		px[i].x = d_clamp(px[i].x, 0, 1);
		px[i].y = d_clamp(px[i].y, 0, 1);
		px[i].z = d_clamp(px[i].z, 0, 1);
	}
	__syncthreads();
}
/**
 * @brief forces the input to be inside of the specified range
 * This function is the same of clamp(), but it increase branch efficency to gain gpu performances
 * 
 * @param a input
 * @param min_ min value of the range
 * @param max_ max value of the range
 * @return float a if it's between min and max, min if a < min, max if a > max
 */
__device__ float d_clamp(float a, float min_, float max_) {
	int minFlag = a < min_;
	int maxFlag = a > max_;
	int flag = minFlag + maxFlag;
	//if(flag > 1) flag = 1; //no way they are both true at the same time IF THE PARAMS ARE CORRECT :<
	return a * (1 - flag) + min_ * minFlag + max_ * maxFlag;
}
/**
 * @brief Applies a smoothstep function
 * This function is the same of smoothstep(), but it increase branch efficency to gain gpu performances
 */
__device__ float d_smoothstep(float a, float b, float u) {
	float t = d_clamp((u - a) / (b - a), 0.0, 1.0);
	return t * t * (3 - 2 * t);
}

/**
 * @brief Remap function to be applied to a single pixel of the subregion in the device's memory inside the llf algorithm
 * It's defined inside the llf's papers https://people.csail.mit.edu/sparis/publi/2011/siggraph/Paris_11_Local_Laplacian_Filters.pdf
 * at page 5 paragaph 4
 * 
 * @param img Image to be remapped on the device's memory
 * @param g0 Reference pixel
 * @param sigma Treshold used by remap function to identify edges and details
 * @param alpha Controls the details level
 * @param beta Controls the tone mapping level
 */
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
/**
 * @brief Applies the remap function to a whole image
 * 
 * @param img Image to be remapped on the device's memory
 * @param g0 Reference pixel
 * @param sigma Treshold used by remap function to identify edges and details
 * @param alpha Controls the details level
 * @param beta Controls the tone mapping level
 */
__device__ void d_remap(Image3 * img, const Pixel3 g0, float sigma, float alpha, float beta){
	uint32_t dim = img -> width * img -> height;
	uint32_t max = dim / blockDim.x;
	Pixel3 *pixels = img -> pixels;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim) //Use multiple threads in the same block to remap the pixels. Since we use this function inside the llf algorithm where each block works on a different pixel, we don't parallelize also for blocks
			pixels[idx] = d_remapSinglePixel(pixels[idx], g0, sigma, alpha, beta);
	}
	__syncthreads();
}