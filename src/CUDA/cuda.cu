#include "cuda.cuh"

#include "../utils/test/testimage.h"

__device__ Pixel3 upsampleConvolveSubtractSinglePixel_shared(Pixel3 *srcPx, uint32_t smallWidth, uint32_t smallHeight, Pixel3 gaussPx, Kernel kernel, uint32_t i, uint32_t j, Pixel3 *convolveWorkingBuffer){
	const int32_t  xstart = -1 * KERNEL_DIMENSION / 2;
	const int32_t  ystart = -1 * KERNEL_DIMENSION / 2;
	
	Pixel3 ups = zero3vect;
	uint32_t idx = threadIdx.x;
	if(idx < (KERNEL_DIMENSION * KERNEL_DIMENSION)){
		uint32_t x = idx % KERNEL_DIMENSION, y = idx / KERNEL_DIMENSION;

		int32_t jy = (j + ystart + y) / 2;
		int32_t ix = (i + xstart + x) / 2;

		int32_t oob = ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight;
		int32_t fi = ix * oob + (i / 2) * (1 - oob), fj = jy * oob + (j / 2) * (1 - oob);

		float kern_elem = kernel[getKernelPosition(x, y)];
		Pixel3 px = d_getPixel3(srcPx, smallWidth, fi, fj);

		vec3MulC(convolveWorkingBuffer[idx], px, kern_elem);
	}

	for(uint32_t stride = 16; stride > 1; stride = stride >> 1){
		if(idx < stride && (idx + stride) < KERNEL_DIMENSION * KERNEL_DIMENSION){
			convolveWorkingBuffer[idx].x += convolveWorkingBuffer[idx + stride].x;
			convolveWorkingBuffer[idx].y += convolveWorkingBuffer[idx + stride].y;
			convolveWorkingBuffer[idx].z += convolveWorkingBuffer[idx + stride].z;
		}
	}
	vec3Add(ups, convolveWorkingBuffer[0], convolveWorkingBuffer[1]);
	vec3Sub(ups, gaussPx, ups);
	return ups;
}
__device__ Pixel3 upsampleConvolveSubtractSinglePixel(Image3 *source, Pixel3 gaussPx, Kernel kernel, uint32_t i, uint32_t j, Pixel3 *convolveWorkingBuffer){
	uint32_t smallWidth = source->width, smallHeight = source->height;
	Pixel3* srcPx = source->pixels;
	const int32_t  xstart = -1 * KERNEL_DIMENSION / 2;
	const int32_t  ystart = -1 * KERNEL_DIMENSION / 2;
	
	Pixel3 ups = zero3vect;
	uint32_t idx = threadIdx.x;
	if(idx < (KERNEL_DIMENSION * KERNEL_DIMENSION)){
		uint32_t x = idx % KERNEL_DIMENSION, y = idx / KERNEL_DIMENSION;

		int32_t jy = (j + ystart + y) / 2;
		int32_t ix = (i + xstart + x) / 2;

		int32_t oob = ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight;
		int32_t fi = ix * oob + (i / 2) * (1 - oob), fj = jy * oob + (j / 2) * (1 - oob);

		float kern_elem = kernel[getKernelPosition(x, y)];
		Pixel3 px = d_getPixel3(srcPx, smallWidth, fi, fj);

		vec3MulC(convolveWorkingBuffer[idx], px, kern_elem);
	}
	for(uint32_t stride = 16; stride > 1; stride = stride >> 1){
		//__syncthreads();
		if(idx < stride && (idx + stride) < KERNEL_DIMENSION * KERNEL_DIMENSION){
			convolveWorkingBuffer[idx].x += convolveWorkingBuffer[idx + stride].x;
			convolveWorkingBuffer[idx].y += convolveWorkingBuffer[idx + stride].y;
			convolveWorkingBuffer[idx].z += convolveWorkingBuffer[idx + stride].z;
		}
	}
	vec3Add(ups, convolveWorkingBuffer[0], convolveWorkingBuffer[1]);
	vec3Sub(ups, gaussPx, ups);
	return ups;
}
__device__ void upsampleConvolveSubtract_fast(Image3 *dest, Image3 *source, Image3 *currentGauss, Kernel kernel, Pixel3 *ds_upsampled){
	uint32_t smallWidth = source->width, smallHeight = source->height;
	uint32_t uppedW = smallWidth << 1;
	uint32_t uppedH = smallHeight << 1;
	uint32_t currentGaussW = currentGauss->width;
	uint32_t yEnd = min(currentGauss->height, uppedH);
	Pixel3 *destPx = dest->pixels, *srcPx = source->pixels, *crtGssPx = currentGauss->pixels;
	if(threadIdx.x == 0){
		dest->width = uppedW;
		dest->height = uppedH;
	}
	uint32_t xEnd = min(currentGaussW, uppedW);
	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;
	
	uint32_t dim = smallWidth * smallHeight;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t x = idx % smallWidth, y = idx / smallWidth;
			d_setPixel3(ds_upsampled, smallWidth, x, y, d_getPixel3(srcPx, smallWidth, x, y));
		}
	}
	__syncthreads();


	dim = xEnd * yEnd;
	max = dim / blockDim.x;
	for(uint32_t li = 0; li <= max; li++){
		uint32_t idx = li * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t i = idx % xEnd, j = idx / xEnd;

			Pixel3 ups = zero3vect;
			for (uint32_t y = 0; y < rows; y++) {
                int32_t jy = (j + ystart + y) / 2;
				for (uint32_t x = 0; x < cols; x++) {
                    int32_t ix = (i + xstart + x) / 2;

					int32_t oob = ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight;
					int32_t fi = ix * oob + (i / 2) * (1 - oob), fj = jy * oob + (j / 2) * (1 - oob);

					float kern_elem = kernel[getKernelPosition(x, y)];
					Pixel3 px = d_getPixel3(ds_upsampled, smallWidth, fi, fj);
					ups.x += px.x * kern_elem;
					ups.y += px.y * kern_elem;
					ups.z += px.z * kern_elem;
				}
			}

			Pixel3 crr = d_getPixel3(crtGssPx, currentGaussW, i, j);
			vec3Sub(crr, crr, ups);
			d_setPixel3(destPx, xEnd, i, j, crr);
		}
	}
	__syncthreads();
}
__device__ void upsampleConvolve(Image3 *dest, Image3 *source, Kernel kernel){
	uint32_t smallWidth = source->width, smallHeight = source->height;
	uint32_t uppedW = smallWidth << 1;
	uint32_t uppedH = smallHeight << 1;
	if(threadIdx.x == 0){
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
	for(uint32_t li = 0; li <= max; li++){
		uint32_t idx = li * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t i = idx % uppedW, j = idx / uppedW;

			Pixel3 c = zero3vect;
			for (uint32_t y = 0; y < rows; y++) {
                int32_t jy = (j + ystart + y) / 2;
				for (uint32_t x = 0; x < cols; x++) {
                    int32_t ix = (i + xstart + x) / 2;

					int32_t oob = ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight;
					int32_t fi = ix * oob + (i / 2) * (1 - oob), fj = jy * oob + (j / 2) * (1 - oob);

					float kern_elem = kernel[getKernelPosition(x, y)];
					Pixel3 px = d_getPixel3(srcPx, smallWidth, fi, fj);
					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
				}
			}
			d_setPixel3(dstPx, uppedW, i, j, c);
		}
	}
	__syncthreads();
}

__device__ void laplacianPyramid(Pyramid laplacian, Pyramid tempGauss, uint8_t nLevels, Kernel filter){
	for(uint8_t i = 0; i < nLevels; i++){
		Image3 *upsampled = laplacian[i];
		upsampleConvolve(upsampled, tempGauss[i + 1], filter);
		//No extra synchtreads needed because there already is one at the end of upsampleConvolve 

		Image3 *current = tempGauss[i];
		Pixel3 *currentPx = current->pixels, *upsampledPx = upsampled->pixels;
		uint32_t yEnd = min(current->height, upsampled->height);
		uint32_t xEnd = min(current->width, upsampled->width);
		uint32_t dim = xEnd * yEnd;
		uint32_t max = dim / blockDim.x;
		for(uint32_t li = 0; li <= max; li++){
			uint32_t idx = li * blockDim.x + threadIdx.x;
			if(idx < dim){
				uint32_t x = idx % xEnd, y = idx / xEnd;
				Pixel3 ups = d_getPixel3(upsampledPx, upsampled->width, x, y);
				Pixel3 crr = d_getPixel3(currentPx, current->width, x, y);
				vec3Sub(crr, crr, ups);
				d_setPixel3(upsampledPx, upsampled->width, x, y, crr);
			}
		}
	}
	//No extra synchtreads needed
	d_imgcpy3(laplacian[nLevels], tempGauss[nLevels]);
}

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

	Pixel3 *destPxs = dest->pixels;
	for(int8_t lev = nLevels; lev > 1; lev--){ //Using dest as a temp buffer
		Image3 *currentLevel = laplacianPyr[lev], *biggerLevel = laplacianPyr[lev - 1];
		Pixel3 *biggerLevelPxs = biggerLevel->pixels;

		upsampleConvolve(dest, currentLevel, lcl_filter);
		//No extra synchtreads needed because there already is one at the end of upsampleConvolve 
		uint32_t sizeUpsampled = min(dest->width, biggerLevel->width) * min(dest->height, biggerLevel->height);
		uint32_t max = sizeUpsampled / blockDim.x;
		for(uint32_t i = 0; i <= max; i++){
			uint32_t px = i * blockDim.x + threadIdx.x;
			if(px < sizeUpsampled)
				vec3Add(biggerLevelPxs[px], destPxs[px], biggerLevelPxs[px]);
		}
		if(threadIdx.x == 0){
			biggerLevel->width = dest->width;
			biggerLevel->height = dest->height; //This could cause disalignment problem
		}
		__syncthreads();
	}
	//No extra synchtreads needed
	Image3 *currentLevel = laplacianPyr[1], *biggerLevel = laplacianPyr[0];
	Pixel3 *biggerLevelPxs = biggerLevel->pixels;

	upsampleConvolve(dest, currentLevel, lcl_filter);
	uint32_t sizeUpsampled = min(dest->width, biggerLevel->width) * min(dest->height, biggerLevel->height);
	max = sizeUpsampled / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t px = i * blockDim.x + threadIdx.x;
		if(px < sizeUpsampled)
			vec3Add(destPxs[px], destPxs[px], biggerLevelPxs[px]);
	}
	__syncthreads();
}

__device__ void downsampleConvolve_shared(Pixel3 *dstPx, Pixel3 *srcPx, uint32_t *width, uint32_t *height, Kernel filter){
	const uint32_t originalW = *width, originalH = *height;
	const uint32_t downW = originalW / 2, downH = originalH / 2;
	*width = downW;
	*height = downH;
	const int32_t startingX = originalW & 1;
	const int32_t startingY = originalH & 1;
	const int8_t  rows = KERNEL_DIMENSION;
	const int8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;

	const int32_t dim = downW * downH; //Small dimensions
	const int32_t max = dim / blockDim.x;
	for(uint32_t li = 0; li <= max; li++){
		int32_t idx = li * blockDim.x + threadIdx.x;
		int32_t i = (idx % downW) * 2 + startingX, j = (idx / downW) * 2 + startingY;
		if(i < originalW && j < originalH){

			Pixel3 c = zero3vect;
			for (uint32_t y = 0; y < rows; y++) {
				int32_t jy = j + (ystart + y) * 2 - startingY;
				for (uint32_t x = 0; x < cols; x++) {
					int32_t ix = i + (xstart + x) * 2 - startingX;

					int32_t oob = ix >= 0 && ix < originalW && jy >= 0 && jy < originalH;
					int32_t fi = ix * oob + (i - startingX) * (1 - oob), fj = jy * oob + (j - startingY) * (1 - oob);

					float kern_elem = filter[getKernelPosition(x, y)];
					Pixel3 px = d_getPixel3(srcPx, originalW, fi, fj);
					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
				}
			}
			d_setPixel3(dstPx, downW, i / 2, j / 2, c);
		}
	}
	__syncthreads();
}
__device__ void downsampleConvolve_fast(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter, Pixel3 *ds_downsampled){
	const uint32_t originalW = *width, originalH = *height;
	const uint32_t downW = originalW / 2, downH = originalH / 2;
	Pixel3 *srcPx = source->pixels;
	Pixel3 *dstPx = dest->pixels;
	*width = downW;
	*height = downH;
	if(threadIdx.x == 0){
		dest->width = downW;
		dest->height = downH;
	}
	uint32_t startingX = originalW & 1;
	uint32_t startingY = originalH & 1;
	
	uint32_t dim = downW * downH;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;

		if(idx < dim){
			uint32_t x = idx % downW, y = idx / downW;
			d_setPixel3(ds_downsampled, downW, x, y, d_getPixel3(srcPx, originalW, (x * 2) + startingX, (y * 2) + startingY));
		}
	}
	__syncthreads();

	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;

	for(uint32_t li = 0; li <= max; li++){
		uint32_t idx = li * blockDim.x + threadIdx.x;

		if(idx < dim){
			uint32_t i = idx % downW, j = idx / downW;
			Pixel3 c = zero3vect;
			for (int32_t y = 0; y < rows; y++) {
				int32_t jy = j + ystart + y;
				for (int32_t x = 0; x < cols; x++) {
					int32_t ix = i + xstart + x;

					int32_t oob = ix >= 0 && ix < downW && jy >= 0 && jy < downH;
					int32_t fi = ix * oob + i * (1 - oob), fj = jy * oob + j * (1 - oob);

					float kern_elem = filter[getKernelPosition(x, y)];
					Pixel3 px = d_getPixel3(ds_downsampled, downW, fi, fj);
					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
				}
			}
			d_setPixel3(dstPx, downW, i, j, c);
		}
	}
	__syncthreads();
}
__device__ void downsampleConvolve(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter){
	const uint32_t originalW = *width, originalH = *height;
	const uint32_t downW = originalW / 2, downH = originalH / 2;
	*width = downW;
	*height = downH;
	if(threadIdx.x == 0){
		dest->width = downW;
		dest->height = downH;
	}
	const int32_t startingX = originalW & 1;
	const int32_t startingY = originalH & 1;
	const int8_t  rows = KERNEL_DIMENSION;
	const int8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;
	Pixel3 *srcPx = source->pixels;
	Pixel3 *dstPx = dest->pixels;

	const int32_t dim = downW * downH; //Small dimensions
	const int32_t max = dim / blockDim.x;
	for(uint32_t li = 0; li <= max; li++){
		int32_t idx = li * blockDim.x + threadIdx.x;
		int32_t i = (idx % downW) * 2 + startingX, j = (idx / downW) * 2 + startingY;
		if(i < originalW && j < originalH){

			Pixel3 c = zero3vect;
			for (uint32_t y = 0; y < rows; y++) {
				int32_t jy = j + (ystart + y) * 2 - startingY;
				for (uint32_t x = 0; x < cols; x++) {
					int32_t ix = i + (xstart + x) * 2 - startingX;

					int32_t oob = ix >= 0 && ix < originalW && jy >= 0 && jy < originalH;
					int32_t fi = ix * oob + (i - startingX) * (1 - oob), fj = jy * oob + (j - startingY) * (1 - oob);

					float kern_elem = filter[getKernelPosition(x, y)];
					Pixel3 px = d_getPixel3(srcPx, originalW, fi, fj); //srcPx[fj * originalW + fi];
					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
				}
			}
			d_setPixel3(dstPx, downW, i / 2, j / 2, c);
		}
	}
	__syncthreads();
}

__device__ void gaussianPyramid_shared(Pixel3 **smallDest, Pixel3 **sourceBigDest, uint32_t *width, uint32_t *height, uint32_t *smallW, uint32_t *smallH, uint8_t nLevels, Kernel d_filter){
	Pixel3 *tempSwap;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		*width = *smallW;
		*height = *smallH;
		downsampleConvolve_shared(*smallDest, *sourceBigDest, smallW, smallH, d_filter);
	//}
	for(uint8_t i = 1; i < nLevels; i++){
		tempSwap = *sourceBigDest;
		*sourceBigDest = *smallDest;
		*smallDest = tempSwap;
		*width = *smallW;
		*height = *smallH;
		downsampleConvolve_shared(*smallDest, *sourceBigDest, smallW, smallH, d_filter);
	}
	//No extra synchtreads needed because there already is one at the end of downsampleConvolve 
}
__device__ void gaussianPyramid_fast(Pyramid d_outPyr, Image3 *d_inImg, uint8_t nLevels, Kernel d_filter, Pixel3 *ds_downsampled){
	d_imgcpy3(d_outPyr[0], d_inImg);
	uint32_t width = d_inImg->width, height = d_inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve_fast(d_outPyr[1], d_inImg, &width, &height, d_filter, ds_downsampled);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve_fast(d_outPyr[i + 1], d_outPyr[i], &width, &height, d_filter, ds_downsampled);
	//No extra synchtreads needed because there already is one at the end of downsampleConvolve 
}
__device__ void __gaussianPyramid_internal(Pyramid d_outPyr, Image3 *d_inImg, uint8_t nLevels, Kernel d_filter){
	d_imgcpy3(d_outPyr[0], d_inImg);
	uint32_t width = d_inImg->width, height = d_inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve(d_outPyr[1], d_inImg, &width, &height, d_filter);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve(d_outPyr[i + 1], d_outPyr[i], &width, &height, d_filter);
	//No extra synchtreads needed because there already is one at the end of downsampleConvolve 
}
__global__ void gaussianPyramid(Pyramid d_outPyr, Image3 *d_inImg, uint8_t nLevels, Kernel d_filter){
	__gaussianPyramid_internal(d_outPyr, d_inImg, nLevels, d_filter);
}



__global__ void __d_llf_internal(Pyramid outputLaplacian, Pyramid gaussPyramid, Image3 *img, uint32_t width, uint32_t height, uint8_t lev, uint32_t subregionDimension, Kernel filter, float sigma, float alpha, float beta, uint16_t elementsNo){
	__shared__ Pixel3 g0;
	__shared__ float lcl_filter[KERNEL_DIMENSION * KERNEL_DIMENSION];
	__shared__ Pixel3 convolveWorkingBuffer[max(MAX_PYR_LAYER * MAX_PYR_LAYER, KERNEL_DIMENSION * KERNEL_DIMENSION)];
	__shared__ Pixel3 convolveWorkingBuffer2[max(MAX_PYR_LAYER * MAX_PYR_LAYER, KERNEL_DIMENSION * KERNEL_DIMENSION)];
	Pixel3 *sourceBigDest = convolveWorkingBuffer, *destSmall = convolveWorkingBuffer2;
	uint32_t dim = KERNEL_DIMENSION * KERNEL_DIMENSION;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
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
		uint32_t x = exIdx % currentW, y = exIdx / currentW;

		int32_t full_res_y = (1 << lev) * y;
		int32_t roi_y0 = full_res_y - subregionDimension;
		int32_t roi_y1 = full_res_y + subregionDimension + 1;
		int32_t base_y = max(0, roi_y0);
		int32_t end_y = min(roi_y1, height);
		int32_t full_res_roi_y = full_res_y - base_y;
		int32_t full_res_roi_yShifted = full_res_roi_y >> lev;

		int32_t full_res_x = (1 << lev) * x;
		int32_t roi_x0 = full_res_x - subregionDimension;
		int32_t roi_x1 = full_res_x + subregionDimension + 1;
		int32_t base_x = max(0, roi_x0);
		int32_t end_x = min(roi_x1, width);
		int32_t full_res_roi_x = full_res_x - base_x;
		int32_t full_res_roi_xShifted = full_res_roi_x >> lev;

		if(threadIdx.x == 0)
			g0 = d_getPixel3(gaussPx, currentW, x, y);
		__syncthreads();

		uint32_t bigW = end_x - base_x, bigH = end_y - base_y;
		uint32_t smallW = bigW, smallH = bigH;
		d_subimage3Remap_shared(sourceBigDest, img, base_x, end_x, base_y, end_y, g0, sigma, alpha, beta);
		gaussianPyramid_shared(&destSmall, &sourceBigDest, &bigW, &bigH, &smallW, &smallH, currentNLevels, lcl_filter);
		Pixel3 gausPx = d_getPixel3(sourceBigDest, bigW, full_res_roi_xShifted, full_res_roi_yShifted);
		Pixel3 outPx = upsampleConvolveSubtractSinglePixel_shared(destSmall, smallW, smallH, gausPx, lcl_filter, full_res_roi_xShifted, full_res_roi_yShifted, sourceBigDest);

		if(threadIdx.x == 0)
			d_setPixel3(outLevPx, outLevW, x, y, outPx);
	}
}

__host__ void llf(Image3 *h_img, float h_sigma, float h_alpha, float h_beta, uint8_t h_nLevels, uint32_t h_nThreads, uint32_t h_elementsNo){
	TimeData timeData;
	TimeCounter passed = 0;

	uint32_t h_width = h_img->width, h_height = h_img->height;
	h_nLevels = min(h_nLevels, MAX_LAYERS);
	h_nLevels = max(h_nLevels, 3);//int(ceil(std::abs(std::log2(min(width, height)) - 3))) + 2;
	Kernel d_filter = createFilterDevice();
	Pyramid d_gaussPyramid = createPyramidDevice(h_width, h_height, h_nLevels);
	Pyramid d_outputLaplacian = createPyramidDevice(h_width, h_height, h_nLevels);

	Image3 *d_img = makeImage3Device(h_width, h_height);
	copyImg3Host2Device(d_img, h_img);
	startTimerCounter(timeData);
	gaussianPyramid<<<1, h_nThreads>>>(d_gaussPyramid, d_img, h_nLevels, d_filter);
	CHECK(cudaDeviceSynchronize());
	stopTimerCounter(timeData, passed);

	startTimerCounter(timeData);
	for(uint8_t h_lev = 0; h_lev < h_nLevels; h_lev++){
		uint32_t h_subregionDimension = 3 * ((1 << (h_lev + 2)) - 1) / 2;
		__d_llf_internal<<<h_elementsNo, h_nThreads>>>(d_outputLaplacian, d_gaussPyramid, d_img, h_width, h_height, h_lev, h_subregionDimension, d_filter, h_sigma, h_alpha, h_beta, h_elementsNo);
		CHECK(cudaDeviceSynchronize());
	}
	d_copyPyrLevel<<<1, h_nThreads>>>(d_outputLaplacian, d_gaussPyramid, h_nLevels);
	CHECK(cudaDeviceSynchronize());
	collapse<<<1, h_nThreads>>>(d_img, d_outputLaplacian, h_nLevels, d_filter);
	CHECK(cudaDeviceSynchronize());
	stopTimerCounter(timeData, passed);
	printff("Total time: %lums\n", passed);

	d_clampImage3<<<(((h_width * h_height) + h_nThreads - 1) / h_nThreads), h_nThreads>>>(d_img);
	CHECK(cudaDeviceSynchronize());

	copyImg3Device2Host(h_img, d_img);

	destroyImage3Device(d_img);
	destroyPyramidDevice(d_gaussPyramid, h_nLevels);
	destroyPyramidDevice(d_outputLaplacian, h_nLevels);
	destroyFilterDevice(d_filter);
}



int main(int argc, char const *argv[]){
	if(argc < 3){
		printff("Usage: %s <number of blocks> <number of threads>\n", argv[0]);
		exit(1);
	}
	int blocksNo = atoi(argv[1]);
	int threadsNo = atoi(argv[2]);
	Image4 *img4 = getStaticImage4();
	Image3 *img = image4to3(img4);
	AlphaMap map = getAlphaMap(img4);
	destroyImage4(&img4);

	llf(img, 0.35, 0.4, 5, 3, threadsNo, blocksNo);

	img4 = image3to4AlphaMap(img, map);
	destroyImage3(&img);
	printStaticImage4(img4);
	destroyImage4(&img4);
}