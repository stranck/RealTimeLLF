#include "cuda.cuh"

#ifdef CUDA_INCLUDE
	#include <cuda.h>
	#include <cuda_runtime.h>
#endif

/**
 * @brief Gets the rendered pixel that can directly be placed into the output laplacian inside the llf's main loop. 
 * The source is a pixel buffer that can also be in shared memory to improve access times
 * 
 * This is a major optimization over the "normal" algorithm:
 * By the paper we have to build the full laplacian pyramid, just to get one pixel from the second lowest layer.
 * Since all layers of a laplacian pyramid are independent from each other, we can just upsample the latest layer of the source gauss pyramid,
 * take the correct pixel and subtract it from the correct pixel of the 2nd latest layer of the gauss pyramid.
 * And not only that! Since the pixel we're taking from the upsampled image depends only by the sorrounding pixels (because we're applying a blur kernel),
 * we can just upsample and convolve the small area of the latest gauss pyramid layer that influences the pixel we're using in the subtraction 
 * 
 * @param srcPx pixel buffer of the image we're going to upsample. This should be the latest layer of the gauss pyramid
 * @param smallWidth width of the source image
 * @param smallHeight height of the source image
 * @param gaussPx the correct pixel of the current gauss pyramid we're gonna use in the subtraction
 * @param kernel blur kernel
 * @param i x coordinates of the pixel on the upsampled image we're gonna use in the subtraction
 * @param j y coordinates of the pixel on the upsampled image we're gonna use in the subtraction
 * @param convolveWorkingBuffer buffer used to temporary store the semi-rendered pixel
 * @return Pixel3 Single pixel we've rendered instead of the complete pyramid, we can place directly on the output laplacian pyramid
 */
__device__ Pixel3 upsampleConvolveSubtractSinglePixel_shared(Pixel3 *srcPx, uint32_t smallWidth, uint32_t smallHeight, Pixel3 gaussPx, Kernel kernel, uint32_t i, uint32_t j, Pixel3 *convolveWorkingBuffer){
	const int32_t  xstart = -1 * KERNEL_DIMENSION / 2;
	const int32_t  ystart = -1 * KERNEL_DIMENSION / 2;
	
	Pixel3 ups = zero3vect;
	uint32_t idx = threadIdx.x;
	if(idx < (KERNEL_DIMENSION * KERNEL_DIMENSION)){ //We suppose we're running this function with at least KERNEL_DIMENSION^2 threads
		uint32_t x = idx % KERNEL_DIMENSION, y = idx / KERNEL_DIMENSION;

		int32_t jy = (j + ystart + y) / 2;
		int32_t ix = (i + xstart + x) / 2;

		int32_t oob = ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight;
		int32_t fi = ix * oob + (i / 2) * (1 - oob), fj = jy * oob + (j / 2) * (1 - oob); //Get the final coordinates for the source pixel

		float kern_elem = kernel[getKernelPosition(x, y)]; //Take the kernel element
		Pixel3 px = d_getPixel3(srcPx, smallWidth, fi, fj); //Take the pixel from the smaller image

		vec3MulC(convolveWorkingBuffer[idx], px, kern_elem); //Render it with the kernel element and save it on the temp buffer. We're gonna add all the pixels together later
	}

	for(uint32_t stride = 16; stride > 1; stride = stride >> 1){ //sum reduction to get the final pixel
		if(idx < stride && (idx + stride) < KERNEL_DIMENSION * KERNEL_DIMENSION){
			convolveWorkingBuffer[idx].x += convolveWorkingBuffer[idx + stride].x;
			convolveWorkingBuffer[idx].y += convolveWorkingBuffer[idx + stride].y;
			convolveWorkingBuffer[idx].z += convolveWorkingBuffer[idx + stride].z;
		}
	}
	vec3Add(ups, convolveWorkingBuffer[0], convolveWorkingBuffer[1]); //last sum reduction
	vec3Sub(ups, gaussPx, ups); //Subtract the blurred pixel we just made from the corresponding pixel of the gaussian pyramid
	return ups;
}
/**
 * @brief Gets the rendered pixel that can directly be placed into the output laplacian inside the llf's main loop
 * 
 * This is a major optimization over the "normal" algorithm:
 * By the paper we have to build the full laplacian pyramid, just to get one pixel from the second lowest layer.
 * Since all layers of a laplacian pyramid are independent from each other, we can just upsample the latest layer of the source gauss pyramid,
 * take the correct pixel and subtract it from the correct pixel of the 2nd latest layer of the gauss pyramid.
 * And not only that! Since the pixel we're taking from the upsampled image depends only by the sorrounding pixels (because we're applying a blur kernel),
 * we can just upsample and convolve the small area of the latest gauss pyramid layer that influences the pixel we're using in the subtraction 
 * 
 * @param source image we're going to upsample. This should be the latest layer of the gauss pyramid
 * @param gaussPx the correct pixel of the current gauss pyramid we're gonna use in the subtraction
 * @param kernel blur kernel
 * @param i x coordinates of the pixel on the upsampled image we're gonna use in the subtraction
 * @param j y coordinates of the pixel on the upsampled image we're gonna use in the subtraction
 * @return Pixel3 Single pixel we've rendered instead of the complete pyramid we can place directly on the output laplacian pyramid
 */
__device__ Pixel3 upsampleConvolveSubtractSinglePixel(Image3 *source, Pixel3 gaussPx, Kernel kernel, uint32_t i, uint32_t j, Pixel3 *convolveWorkingBuffer){
	upsampleConvolveSubtractSinglePixel_shared(source->pixels, source->width, source->height, gaussPx, kernel, i, j, convolveWorkingBuffer);
}
/**
 * @brief Upsamples an image by duplicating it in size and the applying a blur kernel to remove the squares at the same time. It also do a laplacian rendering at the same time
 * Each gpu thread will render a bounch of pixel independently from each other. Unlike upsampleConvolve_cuda we use a temp buffer that should be stored in shared memory to improve access times
 * 
 * This is a first major optimization to the normal upsample convolve described by the llf paper:
 * Instead of building the whole laplacian pyramid, we just upsample and subtract the last layer of the gaussian pyramid
 * 
 * @param dest Destination image
 * @param source Source image 
 * @param kernel Blur kernel
 * @param ds_upsampled Temp buffer that should be located in shared memory to improve access times
 */
__device__ void upsampleConvolveSubtract_fast(Image3 *dest, Image3 *source, Image3 *currentGauss, Kernel kernel, Pixel3 *ds_upsampled){
	uint32_t smallWidth = source->width, smallHeight = source->height;
	uint32_t uppedW = smallWidth << 1;
	uint32_t uppedH = smallHeight << 1;
	uint32_t currentGaussW = currentGauss->width;
	uint32_t yEnd = d_minU32(currentGauss->height, uppedH);
	Pixel3 *destPx = dest->pixels, *srcPx = source->pixels, *crtGssPx = currentGauss->pixels;
	if(threadIdx.x == 0){ //Only one thread should write the new image's dimension
		dest->width = uppedW;
		dest->height = uppedH;
	}
	uint32_t xEnd = d_minU32(currentGaussW, uppedW);
	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;
	
	uint32_t dim = smallWidth * smallHeight;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){ //Use multiple threads to copy the image to the temp buffer in shared memory
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t x = idx % smallWidth, y = idx / smallWidth;
			d_setPixel3(ds_upsampled, smallWidth, x, y, d_getPixel3(srcPx, smallWidth, x, y));
		}
	}
	__syncthreads();


	dim = xEnd * yEnd;
	max = dim / blockDim.x;
	for(uint32_t li = 0; li <= max; li++){ //Each thread is going to handle a different "random" pixel, since there are no dependences between them
		uint32_t idx = li * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t i = idx % xEnd, j = idx / xEnd; //For each pixel C of the image

			Pixel3 ups = zero3vect;
			for (uint32_t y = 0; y < rows; y++) {
                int32_t jy = (j + ystart + y) / 2;
				for (uint32_t x = 0; x < cols; x++) { //For each pixel of the kernel square surrounding C 
                    int32_t ix = (i + xstart + x) / 2; //Half the coordinates to use them on the original smaller image

					int32_t oob = ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight; //Check if we're out of bounds
					int32_t fi = ix * oob + (i / 2) * (1 - oob), fj = jy * oob + (j / 2) * (1 - oob); //Obtain the final coordinates 

					float kern_elem = kernel[getKernelPosition(x, y)]; //Take the kernel element
					Pixel3 px = d_getPixel3(ds_upsampled, smallWidth, fi, fj); //Take the pixel from the smaller image
					ups.x += px.x * kern_elem;
					ups.y += px.y * kern_elem;
					ups.z += px.z * kern_elem; //Apply the kernel element
				}
			}

			Pixel3 crr = d_getPixel3(crtGssPx, currentGaussW, i, j); //Get the corresponding pixel on the gaussian level
			vec3Sub(crr, crr, ups); //Subtract it to apply laplacian
			d_setPixel3(destPx, xEnd, i, j, crr); //Apply the blurred and subtracted pixel C to the upsized image
		}
	}
	__syncthreads();
}
/**
 * @brief Upsamples an image by duplicating it in size and the applying a blur kernel to remove the squares at the same time
 * Each gpu thread will render a bounch of pixel independently from each other
 * 
 * This will save an extra copy of the whole upsized image and the need of an extra temp buffer
 * 
 * @param dest Destination image
 * @param source Source image 
 * @param kernel Blur kernel
 */
__device__ void upsampleConvolve_cuda(Image3 *dest, Image3 *source, Kernel kernel){
	uint32_t smallWidth = source->width, smallHeight = source->height;
	uint32_t uppedW = smallWidth << 1;
	uint32_t uppedH = smallHeight << 1;
	if(threadIdx.x == 0){ //Only one thread should write the new image's dimension
		dest->width = uppedW;
		dest->height = uppedH;
	}
	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;
	Pixel3 *srcPx = source->pixels;
	Pixel3 *dstPx = dest->pixels;
	
	uint32_t dim = uppedW * uppedH;
	uint32_t max = dim / blockDim.x;
	for(uint32_t li = 0; li <= max; li++){ //Each thread is going to handle a different "random" pixel, since there are no dependences between them
		uint32_t idx = li * blockDim.x + threadIdx.x;
		if(idx < dim){ 
			uint32_t i = idx % uppedW, j = idx / uppedW; //For each pixel C of the image

			Pixel3 c = zero3vect;
			for (uint32_t y = 0; y < rows; y++) {
                int32_t jy = (j + ystart + y) / 2;
				for (uint32_t x = 0; x < cols; x++) { //For each pixel of the kernel square surrounding C 
                    int32_t ix = (i + xstart + x) / 2; //Half the coordinates to use them on the original smaller image

					int32_t oob = ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight; //Check if we're out of bounds
					int32_t fi = ix * oob + (i / 2) * (1 - oob), fj = jy * oob + (j / 2) * (1 - oob); //Obtain the final coordinates 

					float kern_elem = kernel[getKernelPosition(x, y)]; //Take the kernel element
					Pixel3 px = d_getPixel3(srcPx, smallWidth, fi, fj); //Take the pixel from the smaller image
					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
				}
			}
			d_setPixel3(dstPx, uppedW, i, j, c); //Apply the blurred pixel C to the downsized image
		}
	}
	__syncthreads();
}

/**
 * @brief Creates a laplacian pyramid starting from a gauss pyramid. It will use multiple threads to subtract the upsampled layer from the gaussian pyramid's layer
 * 
 * Each single layer of a laplacian pyramid is defined as follows:
 * lapl[nLevels] = gauss[nLevels]
 * lapl[n] = gauss[n] - upsample(gauss[n + 1])
 * where 
 * lapl[0] has the original image's dimension
 * and lapl[nLevels] is the smallest image in the pyramid
 * (so, the index is inversely proportional to the size of the layer)
 * 
 * @param laplacian output laplacian pyramid
 * @param tempGauss source gauss pyramid
 * @param nLevels number of layers of both pyramids
 * @param filter blur kernel to be used inside the convolve functions
 */
__device__ void laplacianPyramid_cuda(Pyramid laplacian, Pyramid tempGauss, uint8_t nLevels, Kernel filter){
	for(uint8_t i = 0; i < nLevels; i++){
		Image3 *upsampled = laplacian[i];
		upsampleConvolve_cuda(upsampled, tempGauss[i + 1], filter); //Upsample the next layer using laplacian[i] as temp buffer
		//No extra synchtreads needed because there already is one at the end of upsampleConvolve_cuda 

		Image3 *current = tempGauss[i];
		Pixel3 *currentPx = current->pixels, *upsampledPx = upsampled->pixels;
		uint32_t yEnd = d_minU32(current->height, upsampled->height);
		uint32_t xEnd = d_minU32(current->width, upsampled->width);
		uint32_t dim = xEnd * yEnd;
		uint32_t max = dim / blockDim.x;
		for(uint32_t li = 0; li <= max; li++){ //Each thread will work on a  different independent pixel
			uint32_t idx = li * blockDim.x + threadIdx.x;
			if(idx < dim){
				uint32_t x = idx % xEnd, y = idx / xEnd;
				Pixel3 ups = d_getPixel3(upsampledPx, upsampled->width, x, y); //Get the upsampled pixel
				Pixel3 crr = d_getPixel3(currentPx, current->width, x, y); //Get the pixel from both upsampled and current gauss layers 
				vec3Sub(crr, crr, ups); //Subtract them
				d_setPixel3(upsampledPx, upsampled->width, x, y, crr); //Store them inside the current laplacian pyramid layer
			}
		}
	}
	//No extra synchtreads needed
	d_imgcpy3(laplacian[nLevels], tempGauss[nLevels]); //Manually copy the smallest layer from the gauss to the laplacian pyramid
}

/**
 * @brief Collapses a laplacian pyramid reconstructing an image using multiple threads to add the image layers
 * 
 * The collapse operations starts from the smallest layer (the one with the greatest index) and proceeds as follows:
 * lapl[n - 1] = lapl[n - 1] + upsample(lapl[n])
 * 
 * @param dest destination image
 * @param laplacianPyr source laplacian pyramid
 * @param nLevels number of layers of the laplacian pyramid
 * @param filter blur kernel for the upsample
 */
__global__ void collapse(Image3 *dest, Pyramid laplacianPyr, uint8_t nLevels, Kernel filter){
	__shared__ float lcl_filter[KERNEL_DIMENSION * KERNEL_DIMENSION];
	uint32_t dim = KERNEL_DIMENSION * KERNEL_DIMENSION;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim)
			lcl_filter[idx] = filter[idx];
	}
	__syncthreads();

	Pixel3 *destPxs = dest->pixels; //Using dest as a temp buffer
	for(int8_t lev = nLevels; lev > 1; lev--){ //For each layer except the last one
		Image3 *currentLevel = laplacianPyr[lev], *biggerLevel = laplacianPyr[lev - 1];
		Pixel3 *biggerLevelPxs = biggerLevel->pixels;

		upsampleConvolve_cuda(dest, currentLevel, lcl_filter); //Upsample the current lapl layer and temp save it inside the dest image
		//No extra synchtreads needed because there already is one at the end of upsampleConvolve_cuda 
		uint32_t sizeUpsampled = d_minU32(dest->width, biggerLevel->width) * d_minU32(dest->height, biggerLevel->height);
		uint32_t max = sizeUpsampled / blockDim.x;
		for(uint32_t i = 0; i <= max; i++){
			uint32_t px = i * blockDim.x + threadIdx.x;
			if(px < sizeUpsampled) //Use mutliple threads to add all the pixels of the upsampled layer with the current layer
				vec3Add(biggerLevelPxs[px], destPxs[px], biggerLevelPxs[px]);
		}
		if(threadIdx.x == 0){ //Only one thread should update the image's dimension
			biggerLevel->width = dest->width;
			biggerLevel->height = dest->height; //This could cause disalignment problem
		}
		__syncthreads();
	}
	//Handle the last layer separately to save one extra copy
	Image3 *currentLevel = laplacianPyr[1], *biggerLevel = laplacianPyr[0];
	Pixel3 *biggerLevelPxs = biggerLevel->pixels;

	upsampleConvolve_cuda(dest, currentLevel, lcl_filter);
	uint32_t sizeUpsampled = d_minU32(dest->width, biggerLevel->width) * d_minU32(dest->height, biggerLevel->height);
	max = sizeUpsampled / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t px = i * blockDim.x + threadIdx.x;
		if(px < sizeUpsampled)
			vec3Add(destPxs[px], destPxs[px], biggerLevelPxs[px]);
	}
	__syncthreads();
}

/**
 * @brief Downsamples an image on the gpu memory by halfing it in size and the applying a blur kernel to remove the gaps at the same time
 * The source and the dest are pixel buffers that should be in shared memory to improve access times
 * This will save an extra copy of the whole downsized image and the need of an extra temp buffer
 * 
 * @param dest destination bigger pixel buffer that should be in shared memory
 * @param source source smaller pixel buffer that should be in shared memory
 * @param width pointer to the width of the source image
 * @param height pointer to the height of the source image
 * @param filter blur kernel
 */
__device__ void downsampleConvolve_shared(Pixel3 *dstPx, Pixel3 *srcPx, uint32_t *width, uint32_t *height, Kernel filter){
	const uint32_t originalW = *width, originalH = *height;
	const uint32_t downW = originalW / 2, downH = originalH / 2;
	*width = downW;
	*height = downH; //Half the image dimensions
	const int32_t startingX = originalW & 1;
	const int32_t startingY = originalH & 1; //If the dimension is odd, we copy only the "middle" pixels. Eg the X: -X-X-
	const int8_t  rows = KERNEL_DIMENSION;
	const int8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;

	const int32_t dim = downW * downH; //Small dimensions
	const int32_t max = dim / blockDim.x;
	for(uint32_t li = 0; li <= max; li++){
		int32_t idx = li * blockDim.x + threadIdx.x;
		int32_t i = (idx % downW) * 2 + startingX, j = (idx / downW) * 2 + startingY; //*2 because we need only half of the pixels
		if(i < originalW && j < originalH){ //For half of the pixels C

			Pixel3 c = zero3vect;
			for (uint32_t y = 0; y < rows; y++) {
				int32_t jy = j + (ystart + y) * 2 - startingY;
				for (uint32_t x = 0; x < cols; x++) {  //For each pixel in a KERNEL_DIMENSION^2 square surrounding C
					int32_t ix = i + (xstart + x) * 2 - startingX;

					int32_t oob = ix >= 0 && ix < originalW && jy >= 0 && jy < originalH; //Check if we're in bounds of the bigger image
					int32_t fi = ix * oob + (i - startingX) * (1 - oob), fj = jy * oob + (j - startingY) * (1 - oob); //Compute the final coordinates

					float kern_elem = filter[getKernelPosition(x, y)];
					Pixel3 px = d_getPixel3(srcPx, originalW, fi, fj); //Take the pixel from the bigger image
					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem; //Apply the kernel element
				}
			}
			d_setPixel3(dstPx, downW, i / 2, j / 2, c); //Apply the blurred pixel C to the downsized image
		}
	}
	__syncthreads();
}
/**
 * @brief Downsamples an image on the gpu memory by halfing it in size and applying a blur kernel to remove the gaps at the same time
 * It uses an extra pixel buffer that should be in shared memory as a temp buffer to reduce access times
 * This will save an extra copy of the whole downsized image and the need of an extra temp buffer
 * 
 * @param dest destination bigger image
 * @param source source smaller image
 * @param width pointer to the width of the source image
 * @param height pointer to the height of the source image
 * @param filter blur kernel
 * @param ds_downsampled Temp buffer that should be located in shared memory to improve access time
 */
__device__ void downsampleConvolve_fast(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter, Pixel3 *ds_downsampled){
	const uint32_t originalW = *width, originalH = *height;
	const uint32_t downW = originalW / 2, downH = originalH / 2;
	Pixel3 *srcPx = source->pixels;
	Pixel3 *dstPx = dest->pixels;
	*width = downW;
	*height = downH; //Half the image dimensions and save both of them in the original ptrs and inside the dest image
	if(threadIdx.x == 0){
		dest->width = downW;
		dest->height = downH;
	}
	uint32_t startingX = originalW & 1;
	uint32_t startingY = originalH & 1; //If the dimension is odd, we copy only the "middle" pixels. Eg the X: -X-X-
	
	uint32_t dim = downW * downH;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x; //Use multiple threads to copy the image to the temp buffer in shared memory

		if(idx < dim){
			uint32_t x = idx % downW, y = idx / downW;
			d_setPixel3(ds_downsampled, downW, x, y, d_getPixel3(srcPx, originalW, (x * 2) + startingX, (y * 2) + startingY));  //*2 to copy only the half of the pixel we need and downsize the image
		}
	}
	__syncthreads();

	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;

	//Apply convolve only on the downsize image
	for(uint32_t li = 0; li <= max; li++){ //For each pixel C of the downsized image
		uint32_t idx = li * blockDim.x + threadIdx.x;

		if(idx < dim){
			uint32_t i = idx % downW, j = idx / downW;
			Pixel3 c = zero3vect;
			for (int32_t y = 0; y < rows; y++) { //For each pixel in a KERNEL_DIMENSION^2 square sorrounding C
				int32_t jy = j + ystart + y;
				for (int32_t x = 0; x < cols; x++) {
					int32_t ix = i + xstart + x;

					int32_t oob = ix >= 0 && ix < downW && jy >= 0 && jy < downH; //Check if we're in bounds of the bigger image
					int32_t fi = ix * oob + i * (1 - oob), fj = jy * oob + j * (1 - oob); //Compute the final coordinates

					float kern_elem = filter[getKernelPosition(x, y)];
					Pixel3 px = d_getPixel3(ds_downsampled, downW, fi, fj); //Take the pixel from the bigger image
					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem; //Apply the kernel element
				}
			}
			d_setPixel3(dstPx, downW, i, j, c); //Apply the blurred pixel C to the downsized image
		}
	}
	__syncthreads();
}
/**
 * @brief Downsamples an image on the gpu memory by halfing it in size and the applying a blur kernel to remove the gaps at the same time
 * This will save an extra copy of the whole downsized image and the need of an extra temp buffer
 * 
 * @param dest destination bigger image that should be in shared memory
 * @param source source smaller image buffer that should be in shared memory
 * @param width pointer to the width of the source image
 * @param height pointer to the height of the source image
 * @param filter blur kernel
 */
__device__ void downsampleConvolve_cuda(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter){
	downsampleConvolve_shared(dest->pixels, source->pixels, width, height, filter);
	if(threadIdx.x == 0){ 
		dest->width = *width;
		dest->height = *height;
	}
}

/**
 * @brief Computes the last two layers of a gaussian pyramid into the two pixel buffer in input
 * This is a special implementation to save memory, so both buffers can be saved on the shared memory and improve access time
 * We need less memory, because we don't save the intermediate layers of the gaussian pyramid, we store only the two layers we need to compute the output pixel of the laplacian pyramid
 * To do that we use both buffers alternating them as both source and dest, swapping them at each layer iteration
 * 
 * This is a major optimization over the "normal" algorithm described by the llf paper 

 * 
 * @param smallDest pointer to the variable holding the pixel buffer that, at the end of the algorithm, will contain the smallest layer of the gaussian pyramid (gauss[nLevels]). At each iteration of the function the buffer held by the variable will be swapped with the one pointed by sourceBigDest
 * @param sourceBigDest pointer to the variable holding the pixel buffer that at the start of the algorithm contains the source image and at the end of the algorithm will contain the second-last smallest layer of the gaussian pyramid (gauss[nLevels - 1]). At each iteration of the function the buffer held by the variable will be swapped with the one pointed by smallDest 
 * @param width pointer to the variable that, at the end of the algorithm, will contain the width of the second-last smaller image (gauss[nLevels - 1].width)
 * @param height pointer to the variable that, at the end of the algorithm, will contain the height of the second-last smaller image (gauss[nLevels - 1].height)
 * @param smallW pointer to the variable that at the start of the algorithm contains the width of the source image and at the end will contain the width of the smaller image (gauss[nLevels].width)
 * @param smallH pointer to the variable that at the start of the algorithm contains the height of the source image and at the end will contain the height of the smaller image (gauss[nLevels].height)
 * @param nLevels number of layers of the pyramid
 * @param d_filter blur kernel
 */
__device__ void gaussianPyramid_shared(Pixel3 **smallDest, Pixel3 **sourceBigDest, uint32_t *width, uint32_t *height, uint32_t *smallW, uint32_t *smallH, uint8_t nLevels, Kernel d_filter){
	Pixel3 *tempSwap;
	*width = *smallW;
	*height = *smallH; //Save the w and h of the origianl source image, so after this iteration they will be the dimensions of the second-last smaller image(gauss[nLevels - 1])
	downsampleConvolve_shared(*smallDest, *sourceBigDest, smallW, smallH, d_filter); //Downsample the source image
	for(uint8_t i = 1; i < nLevels; i++){ //Per each layer
		tempSwap = *sourceBigDest;
		*sourceBigDest = *smallDest;
		*smallDest = tempSwap;  //Swap the source/dest buffers, so we save an extra copy and so sourceBigDest will always hold the second-last smaller image(gauss[nLevels - 1])
		*width = *smallW;
		*height = *smallH; //Save the w and h of the origianl source image, so after this iteration they will be the dimensions of the second-last smaller image(gauss[nLevels - 1])
		downsampleConvolve_shared(*smallDest, *sourceBigDest, smallW, smallH, d_filter); //Downsample the source image
		//Due to buffer swapping, after each for iteration we'll have the second-last smaller image(gauss[nLevels - 1]) in sourceBigDest and the smallest layer of the gaussian pyramid (gauss[nLevels]) in smallDest
		//Because we previously saved w and h and downsampleConvolve updates the dimensions of the variable we've passed to it, after each iteration we'll also have the dimensions of the second-last smaller image(gauss[nLevels - 1]) in width and height, and the dimensions of the smallest layer of the gaussian pyramid (gauss[nLevels]) in smallW and smallH
	}
	//No extra synchtreads needed because there already is one at the end of downsampleConvolve_cuda 
}
/**
 * @brief Creates a gaussian pyramid on the devce's memory starting from a source image
 * It uses an extra pixel buffer that should be in shared memory to improve access times
 * 
 * Each single layer of a gaussian pyramid is defined as follows:
 * gauss[0] = sourceImg
 * gauss[n] = downsample(gauss[n - 1])
 * 
 * @param d_outPyr output gaussian pyramid
 * @param d_inImg source image
 * @param nLevels number of layers of the pyramid
 * @param d_filter blur kernel
 * @param ds_downsampled Temp buffer that should be located in shared memory to improve access time
 */
__device__ void gaussianPyramid_fast(Pyramid d_outPyr, Image3 *d_inImg, uint8_t nLevels, Kernel d_filter, Pixel3 *ds_downsampled){
	d_imgcpy3(d_outPyr[0], d_inImg); //Copy the first layer
	uint32_t width = d_inImg->width, height = d_inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve_fast(d_outPyr[1], d_inImg, &width, &height, d_filter, ds_downsampled);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve_fast(d_outPyr[i + 1], d_outPyr[i], &width, &height, d_filter, ds_downsampled); //Downsample the current layer and save it into the next one
	//No extra synchtreads needed because there already is one at the end of downsampleConvolve_cuda 
}
/**
 * @brief Creates a gaussian pyramid on the device's memory starting from a source image
 * 
 * This is the internal function that will be called from gaussianPyramid_cuda
 * 
 * Each single layer of a gaussian pyramid is defined as follows:
 * gauss[0] = sourceImg
 * gauss[n] = downsample(gauss[n - 1])
 * 
 * @param d_outPyr output gaussian pyramid
 * @param d_inImg source image
 * @param nLevels number of layers of the pyramid
 * @param d_filter blur kernel
 */
__device__ void __gaussianPyramid_internal(Pyramid d_outPyr, Image3 *d_inImg, uint8_t nLevels, Kernel d_filter){
	d_imgcpy3(d_outPyr[0], d_inImg); //Copy the first layer
	uint32_t width = d_inImg->width, height = d_inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve_cuda(d_outPyr[1], d_inImg, &width, &height, d_filter);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve_cuda(d_outPyr[i + 1], d_outPyr[i], &width, &height, d_filter); //Downsample the current layer and save it into the next one
	//No extra synchtreads needed because there already is one at the end of downsampleConvolve_cuda 
}
/**
 * @brief Creates a gaussian pyramid on the device's memory starting from a source image
 * 
 * Each single layer of a gaussian pyramid is defined as follows:
 * gauss[0] = sourceImg
 * gauss[n] = downsample(gauss[n - 1])
 * 
 * @param d_outPyr output gaussian pyramid
 * @param d_inImg source image
 * @param nLevels number of layers of the pyramid
 * @param d_filter blur kernel
 */
__global__ void gaussianPyramid_cuda(Pyramid d_outPyr, Image3 *d_inImg, uint8_t nLevels, Kernel d_filter){
	__gaussianPyramid_internal(d_outPyr, d_inImg, nLevels, d_filter);
}

/**
 * @brief internal function of the llf algorithm that runs on the device and renders on the gpu a whole layer of the input gaussian pyramid. 
 * Each block will render a batch of pixel, and the rendering of each single pixel is handled by multiple threads
 * 
 * @param outputLaplacian Destination laplacian pyramid that this rendering will fill and is going to be collapsed
 * @param gaussPyramid Source gaussian pyramid
 * @param img Source image
 * @param width Width of the source image
 * @param height Height of the source image
 * @param lev Current level we're working on
 * @param subregionDimension Dimension of the subregion of the current layer we're working on
 * @param filter blur kernel for convolving
 * @param sigma Treshold used by remap function to identify edges and details
 * @param alpha Controls the details level
 * @param beta Controls the tone mapping level
 * @param elementsNo Number of blocks (AKA number of pixels we're rendering "at the same time") used to launch this function
 */
__global__ void __d_llf_internal(Pyramid outputLaplacian, Pyramid gaussPyramid, Image3 *img, uint32_t width, uint32_t height, uint8_t lev, uint32_t subregionDimension, Kernel filter, float sigma, float alpha, float beta, uint16_t elementsNo){
	__shared__ Pixel3 g0; //Shared current pixel
	__shared__ float lcl_filter[KERNEL_DIMENSION * KERNEL_DIMENSION]; //Local blur kernel buffer in shared memory
	__shared__ Pixel3 convolveWorkingBuffer [llf_max(MAX_PYR_LAYER * MAX_PYR_LAYER, KERNEL_DIMENSION * KERNEL_DIMENSION)]; //Buffer #1 in shared memory to create the last two layers of the gauss pyramid improving access times
	__shared__ Pixel3 convolveWorkingBuffer2[llf_max(MAX_PYR_LAYER * MAX_PYR_LAYER, KERNEL_DIMENSION * KERNEL_DIMENSION)]; //Buffer #2 in shared memory to create the last two layers of the gauss pyramid improving access times
	Pixel3 *sourceBigDest = convolveWorkingBuffer, *destSmall = convolveWorkingBuffer2; //We make local pointers to these buffers, so we can swap them in the gaussianPyramid process
	uint32_t dim = KERNEL_DIMENSION * KERNEL_DIMENSION;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){ //Locally copy the blur kernel to shared memory to improve access times
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim)
			lcl_filter[idx] = filter[idx];
	}
	__syncthreads();

	Image3 *currentGaussLevel = gaussPyramid[lev];
	Image3 *outLev = outputLaplacian[lev];
	uint8_t currentNLevels = lev + 1;
	uint32_t outLevW = outLev->width;
	Pixel3 *outLevPx = outLev->pixels, *gaussPx = currentGaussLevel->pixels;

	//uint32_t x = blockIdx.x, y = blockIdx.y;
	uint32_t currentW = currentGaussLevel->width;
	uint32_t exDim = currentW * currentGaussLevel->height;
	uint32_t exMax = exDim / elementsNo;
	for(uint32_t exId = 0; exId <= exMax; exId++){ 
		uint32_t exIdx = exId * elementsNo + blockIdx.x;
		if(exIdx >= exDim) return;
		uint32_t x = exIdx % currentW, y = exIdx / currentW; //For each pixel

		int32_t full_res_y = (1 << lev) * y;
		int32_t roi_y0 = full_res_y - subregionDimension;
		int32_t roi_y1 = full_res_y + subregionDimension + 1;
		int32_t base_y = d_maxI32(0, roi_y0);
		int32_t end_y = d_minI32(roi_y1, height);
		int32_t full_res_roi_y = full_res_y - base_y;
		int32_t full_res_roi_yShifted = full_res_roi_y >> lev;

		int32_t full_res_x = (1 << lev) * x;
		int32_t roi_x0 = full_res_x - subregionDimension;
		int32_t roi_x1 = full_res_x + subregionDimension + 1;
		int32_t base_x = d_maxI32(0, roi_x0);
		int32_t end_x = d_minI32(roi_x1, width);
		int32_t full_res_roi_x = full_res_x - base_x;
		int32_t full_res_roi_xShifted = full_res_roi_x >> lev;

		if(threadIdx.x == 0) //If we're thread 0 we load the pixel g0 from the gaussian pyramid
			g0 = d_getPixel3(gaussPx, currentW, x, y);
		__syncthreads();

		uint32_t bigW = end_x - base_x, bigH = end_y - base_y;
		uint32_t smallW = bigW, smallH = bigH;
		d_subimage3Remap_shared(sourceBigDest, img, base_x, end_x, base_y, end_y, g0, sigma, alpha, beta); //Cuts a subregion from the original image and store them inside the first buffer. We also remap the subregion at the same time
		gaussianPyramid_shared(&destSmall, &sourceBigDest, &bigW, &bigH, &smallW, &smallH, currentNLevels, lcl_filter); //Starting from the subregion, compute the smallest and the second smallest layer of the gaussian pyramid using the two buffers in shared memory to improve access times
		Pixel3 gausPx = d_getPixel3(sourceBigDest, bigW, full_res_roi_xShifted, full_res_roi_yShifted); //Get the pixel we're gonna subtract in the reduced laplacian pyramid from the second last smaller level we've just render
		Pixel3 outPx = upsampleConvolveSubtractSinglePixel_shared(destSmall, smallW, smallH, gausPx, lcl_filter, full_res_roi_xShifted, full_res_roi_yShifted, sourceBigDest); //Upsample the smallest layer of the gaussian pyramid we've just rendered and subtract the previous pixel to get the output we're gonna place in the outputLaplacianPyramid

		if(threadIdx.x == 0) //Store the pixel we've just rendered in the output laplacian pyramid
			d_setPixel3(outLevPx, outLevW, x, y, outPx);
	}
}

/**
 * @brief Apply the local laplacian filter over one image using cuda
 * 
 * Each block is in charge of working on a batch of pixels taken from the current layer of the gaussian pyramid
 * The rendering of each single pixel uses multiple thread. We parallelize the rendering of each pixel and we render more pixel at the same time
 * We also parallelize the creation of the first gaussian pyramid and the collapse function
 * 
 * Local laplacian filter works as follows:
 * - Create a gaussian pyramid starting from the source image
 * - For each pixel, for each layer of the gauss pyramid:
 * -- take the current pixel G0 from the original gaussian pyramid
 * -- cut a subregion R0 around G0 from the source image with a dimension proportional to the layer's dimension near the pixel
 * -- apply a remap function to R0 using G0 as reference
 * -- compute only the last two layers of a gaussian pyramid over R0
 * -- get the pixel GAUSSPX at the correct coordinates respect to the original pixel from the second-last layer of the gaussian pyramid we've just computed
 * -- instead of creating a whole laplacian pyramid, render only the pixel placed at the same coordinates of GAUSSPX, using GAUSSPX and the last layer of the gaussian pyramid we've just computed
 * -- copy the pixel we've just rendered to the current layer of the output laplacian pyramid
 * - copy the smallest layer of the gaussian pyramid over the output laplacian pyramid
 * - collapse the output laplacian pyramid over the destination image
 * - clamp the destination image
 * 
 * @param h_img source AND destination image. The content of this image is going to be overwritten after the algorithm completes!
 * @param h_sigma Treshold used by remap function to identify edges and details
 * @param h_alpha Controls the details level
 * @param h_beta Controls the tone mapping level
 * @param h_nLevels Number of layers of the pyramids
 * @param h_cudaBuffers Pre-allocated data structures that will be used during the processing
 */
__host__ void llf(Image3 *h_img, float h_sigma, float h_alpha, float h_beta, uint8_t h_nLevels, uint32_t h_nThreads, uint32_t h_elementsNo, WorkingBuffers *h_cudaBuffers){
	TimeData timeData;
	TimeCounter passed = 0;

	uint32_t h_width = h_img->width, h_height = h_img->height;
	h_nLevels = llf_min(h_nLevels, MAX_LAYERS); //Clamps the number of levels
	h_nLevels = llf_max(h_nLevels, 2);//int(ceil(std::abs(std::log2(llf_min(width, height)) - 3))) + 2;
	Kernel d_filter = h_cudaBuffers->d_filter;
	Pyramid d_gaussPyramid = h_cudaBuffers->d_gaussPyramid;
	Pyramid d_outputLaplacian = h_cudaBuffers->d_outputLaplacian;

	Image3 *d_img = h_cudaBuffers->d_img;
	copyImg3Host2Device(d_img, h_img); //Copy the source image to the device
	startTimerCounter(timeData);
	gaussianPyramid_cuda<<<1, h_nThreads>>>(d_gaussPyramid, d_img, h_nLevels, d_filter); //Creates a gaussian pyramid starting from the source image
	CHECK(cudaDeviceSynchronize());
	stopTimerCounter(timeData, passed);

	startTimerCounter(timeData);
	for(uint8_t h_lev = 0; h_lev < h_nLevels; h_lev++){ //For each layer of the gaussian pyramid
		uint32_t h_subregionDimension = 3 * ((1 << (h_lev + 2)) - 1) / 2; //Get the subregion dimension of that layer
		__d_llf_internal<<<h_elementsNo, h_nThreads>>>(d_outputLaplacian, d_gaussPyramid, d_img, h_width, h_height, h_lev, h_subregionDimension, d_filter, h_sigma, h_alpha, h_beta, h_elementsNo); //Render the level working on the multiple pixels at the same time, and using multiple thread to render each pixel
		CHECK(cudaDeviceSynchronize());
	}
	d_copyPyrLevel<<<1, h_nThreads>>>(d_outputLaplacian, d_gaussPyramid, h_nLevels); //Copy the smallest layer of the gauss pyramid over the output laplacian pyramid using multiple threads
	CHECK(cudaDeviceSynchronize());
	collapse<<<1, h_nThreads>>>(d_img, d_outputLaplacian, h_nLevels, d_filter); //Collapse the output laplacian pyramid over the dest image using multiple threads
	CHECK(cudaDeviceSynchronize());
	stopTimerCounter(timeData, passed);
	#ifdef SHOW_TIME_STATS
		printff("Total time: %lums\n", passed);
	#endif
	d_clampImage3<<<(((h_width * h_height) + h_nThreads - 1) / h_nThreads), h_nThreads>>>(d_img); //Clamp the dest image to put all the pixel in the [0;1] bounds using multiple threads and blocks
	CHECK(cudaDeviceSynchronize());

	copyImg3Device2Host(h_img, d_img); //Copy back the rendered image to the host
}

/**
 * @brief allocates the data structures needed for cuda llf's processing
 * 
 * @param h_cudaBuffers non-allocated data structures
 * @param h_width width of the pyramids
 * @param h_height height of the pyramids
 * @param h_nLevels number of layers of the pyramids
 */
__host__ void initWorkingBuffers(WorkingBuffers *h_cudaBuffers, uint32_t h_width, uint32_t h_height, uint8_t h_nLevels){
	h_cudaBuffers->d_outputLaplacian = createPyramidDevice(h_width, h_height, h_nLevels);
	h_cudaBuffers->d_gaussPyramid = createPyramidDevice(h_width, h_height, h_nLevels);
	h_cudaBuffers->d_img = makeImage3Device(h_width, h_height);
	h_cudaBuffers->d_filter = createFilterDevice();
}
/**
 * @brief destroy the data structures needed for cuda llf's processing
 * 
 * @param h_cudaBuffers allocated data structures
 * @param h_nLevels number of layers of the already allocated pyramids
 */
__host__ void destroyWorkingBuffers(WorkingBuffers *h_cudaBuffers, uint8_t h_nLevels){
	destroyImage3Device(h_cudaBuffers->d_img);
	destroyPyramidDevice(h_cudaBuffers->d_gaussPyramid, h_nLevels);
	destroyPyramidDevice(h_cudaBuffers->d_outputLaplacian, h_nLevels);
	destroyFilterDevice(h_cudaBuffers->d_filter);
}