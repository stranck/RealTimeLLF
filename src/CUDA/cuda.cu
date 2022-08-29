#include "cuda.cuh"

#include "../utils/test/testimage.h"

__device__ void upsampleConvolve(Image3 *dest, Image3 *source, Kernel kernel){
	uint32_t smallWidth = source->width, smallHeight = source->height;
	uint32_t uppedW = smallWidth << 1;
	uint32_t uppedH = smallHeight << 1;
	if(threadIdx.x == 0){
		dest->width = uppedW;
		dest->height = uppedH;
	}
	__syncthreads();
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

					double kern_elem = kernel[getKernelPosition(x, y)];
					Pixel3 px = d_getPixel3(srcPx, smallWidth, fi, fj); //srcPx[fj * smallWidth + fi];
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

__device__ void downsampleConvolve(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter){
	const uint32_t originalW = *width, originalH = *height;
	const uint32_t downW = originalW / 2, downH = originalH / 2;
	//printf("a\n");
	//printf("Addr: *width: 0x%016llx  *height: 0x%016llx  *dW: 0x%016llx  *dH: 0x%016llx\n", width, height, &(dest->width), &(dest->height));
	//printf("Entering downsampleConvolve Tid: %d   orgW: %d   orgH: %d   downW: %d   downH: %d - ptrW: %d   ptrH: %d   dW: %d   dH: %d\n", threadIdx.x, originalW, originalH, downW, downH, *width, *height, dest->width, dest->height);
	*width = downW;
	*height = downH;
	if(threadIdx.x == 0){
		dest->width = downW;
		dest->height = downH;
	}
	__syncthreads();
	//printf("Resized dimensions: Tid: %d   orgW: %d   orgH: %d   downW: %d   downH: %d - ptrW: %d   ptrH: %d   dW: %d   dH: %d\n", threadIdx.x, originalW, originalH, downW, downH, *width, *height, dest->width, dest->height);
	const int32_t startingX = originalW & 1;
	const int32_t startingY = originalH & 1;
	const int8_t  rows = KERNEL_DIMENSION;
	const int8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;
	Pixel3 *srcPx = source->pixels;
	Pixel3 *dstPx = dest->pixels;
	//printf("b\n");

	const int32_t dim = downW * downH; //Small dimensions
	const int32_t max = dim / blockDim.x;
	//printf("Entering loop Tid: %d\n", threadIdx.x);
	for(uint32_t li = 0; li <= max; li++){
		int32_t idx = li * blockDim.x + threadIdx.x;
		int32_t i = (idx % downW) * 2 + startingX, j = (idx / downW) * 2 + startingY;
		//printf("IDX: % 6d            i:% 4d            j:% 4d            im:% 4d            jd:% 4d            downW: %d   downH: %d   orgW: %d   orgH: %d   bool: %u%u\n", idx, i, j, (idx % downW), (idx / downW), downW, downH, originalW, originalH, i < originalH, );
		//if(threadIdx.x == 1 && li > 53300) printf("[%d; %d * %d + %d] Starting loop j: %d   i: %d   originalW: %d   originalH: %d   downW: %d   downH: %d\n", idx, li, blockDim.x, threadIdx.x, originalW, originalH, downW, downH);
		if(i < originalW && j < originalH){

	//for (uint32_t j = startingY; j < originalH; j += 2) {
	//	for (uint32_t i = startingX; i < originalW; i += 2) {

			Pixel3 c = zero3vect;
			for (uint32_t y = 0; y < rows; y++) {
				int32_t jy = j + (ystart + y) * 2 - startingY;
				for (uint32_t x = 0; x < cols; x++) {
					int32_t ix = i + (xstart + x) * 2 - startingX;

					int32_t oob = ix >= 0 && ix < originalW && jy >= 0 && jy < originalH;
					int32_t fi = ix * oob + (i - startingX) * (1 - oob), fj = jy * oob + (j - startingY) * (1 - oob);

					double kern_elem = filter[getKernelPosition(x, y)];
					//if(threadIdx.x == 1 && li > 53300) printf("[%d; %d * %d + %d] -> Fi: %d   Fj: %d   originalW: %d   originalH: %d   oob: %d   ix: %d   i: %d   startingX: %d   jy: %d   j: %d   startingY: %d\n", idx, li, blockDim.x, threadIdx.x, fi, fj, originalW, originalH, oob, ix, i, startingX, jy, j, startingY);
					Pixel3 px = d_getPixel3(srcPx, originalW, fi, fj); //srcPx[fj * originalW + fi];
					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;

					/*int32_t ix = i + (xstart + x) * 2 - startingX;

					if (ix >= 0 && ix < originalW && jy >= 0 && jy < originalH) {
						double kern_elem = filter[getKernelPosition(x, y)];
						Pixel3 px = d_getPixel3(srcPx, originalW, ix, jy);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					} else {
						
						double kern_elem = filter[getKernelPosition(x, y)];
						//printf("[%d; %d * %d + %d] -> i: %d  j: %d  sX: %d  sY: %d  orgW: %d  orgH: %d  calc: %d  Addr: 0x%016llx\n", idx, li, blockDim.x, threadIdx.x, i, j, startingX, startingY, originalW, originalH, (j - startingY) * originalW + (i - startingX), srcPx);
						Pixel3 px = srcPx[(j - startingY) * (originalW) + (i - startingX)]; //d_getPixel3(srcPx, originalW, i - startingX, j - startingY);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					}*/
				}
			}
			//c.x = 0; c.y = 0; c.z = 0;
			//if(threadIdx.x == 1 && li > 53300) printf("[%d; %d * %d + %d] <- j: %d   i: %d   j2: %d   i2: %d   originalW: %d   originalH: %d\n", idx, li, blockDim.x, threadIdx.x, j, i, j/2, i/2, originalW, originalH);
			d_setPixel3(dstPx, downW, i / 2, j / 2, c);
		}
	}
	//printf("Exiting loop Tid: %d\n", threadIdx.x);
	__syncthreads();
}

__global__ void gaussianPyramid(Pyramid d_outPyr, Image3 *d_inImg, uint8_t nLevels, Kernel d_filter){
	//printf("Tid: %d\n", threadIdx.x);
	__gaussianPyramid_internal(d_outPyr, d_inImg, nLevels, d_filter);
	//printf("gaussianPyramid done\n");
}
__device__ void __gaussianPyramid_internal(Pyramid d_outPyr, Image3 *d_inImg, uint8_t nLevels, Kernel d_filter){
	d_imgcpy3(d_outPyr[0], d_inImg);
	uint32_t width = d_inImg->width, height = d_inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve(d_outPyr[1], d_inImg, &width, &height, d_filter);
		//Image3 *h_imgDst = d_outPyr[1];
		//printf("%f %f %f %f %f\n", h_imgDst->pixels[0].y, h_imgDst->pixels[0].y, h_imgDst->pixels[1].y, h_imgDst->pixels[2].y, h_imgDst->pixels[3].y, h_imgDst->pixels[4].y);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve(d_outPyr[i + 1], d_outPyr[i], &width, &height, d_filter);
	//No extra synchtreads needed because there already is one at the end of downsampleConvolve 
}

__global__ void laplacianPyramidTest(Pyramid laplacian, Pyramid tempGauss, uint8_t nLevels, Kernel filter){
	laplacianPyramid(laplacian, tempGauss, nLevels, filter);
}
__device__ void laplacianPyramid(Pyramid laplacian, Pyramid tempGauss, uint8_t nLevels, Kernel filter){
	for(uint8_t i = 0; i < nLevels; i++){
		Image3 *upsampled = laplacian[i];
		upsampleConvolve(upsampled, tempGauss[i + 1], filter);
		//No extra synchtreads needed because there already is one at the end of upsampleConvolve 

		Image3 *current = tempGauss[i];
		//TODO Check if min macro works fine for cuda
		Pixel3 *currentPx = current->pixels, *upsampledPx = upsampled->pixels;
		uint32_t yEnd = min(current->height, upsampled->height);
		uint32_t xEnd = min(current->width, upsampled->width);
		uint32_t dim = xEnd * yEnd;
		uint32_t max = dim / blockDim.x;
		for(uint32_t li = 0; li <= max; li++){
			uint32_t idx = li * blockDim.x + threadIdx.x;
			if(idx < dim){
				Pixel3 ups = upsampledPx[idx];
				Pixel3 crr = currentPx[idx];

				upsampledPx[idx] = vec3Sub(crr, ups, Pixel3);
			}
		}
		__syncthreads();
	}
	//No extra synchtreads needed
	d_imgcpy3(laplacian[nLevels], tempGauss[nLevels]);
}

__global__ void collapse(Image3 *dest, Pyramid laplacianPyr, uint8_t nLevels, Kernel filter){
	__shared__ double lcl_filter[KERNEL_DIMENSION * KERNEL_DIMENSION];
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
				biggerLevelPxs[px] = vec3Add(destPxs[px], biggerLevelPxs[px], Pixel3);
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
			destPxs[px] = vec3Add(destPxs[px], biggerLevelPxs[px], Pixel3);
	}
	__syncthreads();
}

#if SYNC_PRIMITIVES_SUPPORTED
__global__ void __d_llf_internal(Pyramid outputLaplacian, Pyramid gaussPyramid, Image3 *img, uint32_t width, uint32_t height, uint8_t lev, uint32_t subregionDimension, Kernel filter, double sigma, double alpha, double beta, PyrBuffer *buffer){
#else
__global__ void __d_llf_internal(Pyramid outputLaplacian, Pyramid gaussPyramid, Image3 *img, uint32_t width, uint32_t height, uint8_t lev, uint32_t subregionDimension, Kernel filter, double sigma, double alpha, double beta, PyrBuffer *buffer, uint8_t elementsNo){
#endif
	__shared__ double lcl_filter[KERNEL_DIMENSION * KERNEL_DIMENSION];
	Image3 *currentGaussLevel = gaussPyramid[lev];
	#if SYNC_PRIMITIVES_SUPPORTED
		uint32_t x = blockIdx.x, y = blockIdx.y;
	#else
		uint32_t currentW = currentGaussLevel->width, currentH = currentGaussLevel->height;
		uint32_t exDim = currentW * currentH;
		uint32_t exMax = exDim / elementsNo;
		for(uint32_t exId = 0; exId <= exMax; exId++){
			uint32_t exIdx = exId * elementsNo + blockIdx.x;
			if(exIdx >= exDim) return;
			uint32_t x = exIdx % currentW, y = exIdx / currentW;
			//printf("llf_internal: block% 4d    thread% 4d    exId% 4d\n", blockIdx.x, threadIdx.x, exIdx);
	#endif

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

	//printf("Copying blur kernel %ux%u\n", x, y);
	uint32_t dim = KERNEL_DIMENSION * KERNEL_DIMENSION;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim)
			lcl_filter[idx] = filter[idx];
	}
	__syncthreads();

	__shared__ Pyramid bufferLaplacianPyramid, bufferGaussPyramid;
	__shared__ Pixel3 g0;
	__shared__ NodeBuffer *node;
	if(threadIdx.x == 0){
		node = d_aquireBuffer(buffer);
		bufferLaplacianPyramid = node->bufferLaplacianPyramid;
		bufferGaussPyramid = node->bufferGaussPyramid;

		g0 = d_getPixel3(currentGaussLevel->pixels, currentGaussLevel->width, x, y);
	}
	__syncthreads();

	//if(exIdx == 78) printf("Idx.x %d    exIdx %d    bufferLapl[0] addr: 0x%012llx\n", blockIdx.x, exIdx, bufferLaplacianPyramid[0]);
	//printf("subimage %ux%u\n", x, y);
	d_subimage3(bufferLaplacianPyramid[0], img, base_x, end_x, base_y, end_y); //Using bufferLaplacianPyramid[0] as temp buffer
	//printf("remap %ux%u\n", x, y);
	d_remap(bufferLaplacianPyramid[0], g0, sigma, alpha, beta);
	uint8_t currentNLevels = lev + 1;
	//printf("remap %ux%u\n", x, y);
	__gaussianPyramid_internal(bufferGaussPyramid, bufferLaplacianPyramid[0], currentNLevels, lcl_filter);
	//printf("laplacian %ux%u\n", x, y);
	laplacianPyramid(bufferLaplacianPyramid, bufferGaussPyramid, currentNLevels, lcl_filter);

	//printf("pixel copy %ux%u\n", x, y);
	if(threadIdx.x == 0){
		Image3 *outLev = outputLaplacian[lev], *crtLev = bufferLaplacianPyramid[lev];
		d_setPixel3(outLev->pixels, outLev->width, x, y, d_getPixel3(crtLev->pixels, crtLev->width, full_res_roi_x >> lev, full_res_roi_yShifted)); //idk why i had to shift those
		
		d_releaseBuffer(node, buffer);
	}
	__syncthreads();
	
	#if !(SYNC_PRIMITIVES_SUPPORTED)
		}
	#endif
}

__host__ void llf(Image3 *h_img, double h_sigma, double h_alpha, double h_beta, uint8_t h_nLevels, uint32_t h_nThreads, uint32_t h_elementsNo){

	size_t rsize = 1024ULL*1024ULL*1024ULL*4ULL;  // allocate 4GB
	CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, rsize));

	uint32_t h_width = h_img->width, h_height = h_img->height;
	h_nLevels = min(h_nLevels, 5);
	h_nLevels = max(h_nLevels, 3);//int(ceil(std::abs(std::log2(min(width, height)) - 3))) + 2;
	print("Creating blur kernel");
	Kernel d_filter = createFilterDevice();
	print("Creating gauss pyr");
	Pyramid d_gaussPyramid = createPyramidDevice(h_width, h_height, h_nLevels);
	print("Creating lapl pyr");
	Pyramid d_outputLaplacian = createPyramidDevice(h_width, h_height, h_nLevels);

	print("Create buffer device");
	PyrBuffer *d_buffer = createBufferDevice(h_elementsNo, (3 * ((1 << (h_nLevels + 1)) - 1)), h_nLevels);

	print("makeimage3");
	Image3 *d_img = makeImage3Device(h_width, h_height);
	print("copyImg3");
	copyImg3Host2Device(d_img, h_img);
	print("FIRST KERNEL");
	gaussianPyramid<<<1, h_nThreads>>>(d_gaussPyramid, d_img, h_nLevels, d_filter);
	CHECK(cudaDeviceSynchronize());

	/*laplacianPyramidTest<<<1, 128>>>(d_outputLaplacian, d_gaussPyramid, h_nLevels, d_filter);
	CHECK(cudaDeviceSynchronize());
	d_copyPyrLevel<<<1, 128>>>(d_outputLaplacian, d_gaussPyramid, h_nLevels);
	CHECK(cudaDeviceSynchronize());
	collapse<<<1, 128>>>(d_img, d_outputLaplacian, h_nLevels, d_filter);
	CHECK(cudaDeviceSynchronize());

	fflush(stdout);

	Image3 *d_blurImg = getImageFromPyramidDevice(d_gaussPyramid, 0);
	//d_clampImage3<<<(((h_width * h_height) + h_nThreads - 1) / h_nThreads), h_nThreads>>>(d_img);
	//CHECK(cudaDeviceSynchronize());
	//d_subimage3Test<<<1, 64>>>(d_img, d_blurImg, 500, 625, 32, 190);
	copyImg3Device2Host(h_img, d_blurImg);*/

	for(uint8_t h_lev = 0; h_lev < h_nLevels; h_lev++){
		printff("Loop %u\n", h_lev);
		uint32_t h_layerW, h_layerH;
		getPyramidDimensionsAtLayer(d_gaussPyramid, h_lev, &h_layerW, &h_layerH);
		dim3 grid(h_layerW, h_layerH);
		uint32_t h_subregionDimension = 3 * ((1 << (h_lev + 2)) - 1) / 2;

		#if SYNC_PRIMITIVES_SUPPORTED
			__d_llf_internal<<<grid, h_nThreads>>>(d_outputLaplacian, d_gaussPyramid, d_img, h_width, h_height, h_lev, h_subregionDimension, d_filter, h_sigma, h_alpha, h_beta, d_buffer);
		#else
			__d_llf_internal<<<h_elementsNo, h_nThreads>>>(d_outputLaplacian, d_gaussPyramid, d_img, h_width, h_height, h_lev, h_subregionDimension, d_filter, h_sigma, h_alpha, h_beta, d_buffer, h_elementsNo);
		#endif
		CHECK(cudaDeviceSynchronize());
	}
	d_copyPyrLevel<<<1, h_nThreads>>>(d_outputLaplacian, d_gaussPyramid, h_nLevels);
	CHECK(cudaDeviceSynchronize());
	collapse<<<1, h_nThreads>>>(d_img, d_outputLaplacian, h_nLevels, d_filter);
	CHECK(cudaDeviceSynchronize());
	d_clampImage3<<<(((h_width * h_height) + h_nThreads - 1) / h_nThreads), h_nThreads>>>(d_img);
	CHECK(cudaDeviceSynchronize());

	copyImg3Device2Host(h_img, d_img);

	destroyBufferDevice(h_elementsNo, h_nLevels, d_buffer);
	destroyImage3Device(d_img);
	destroyPyramidDevice(d_gaussPyramid, h_nLevels);
	destroyPyramidDevice(d_outputLaplacian, h_nLevels);
	destroyFilterDevice(d_filter);
}

uint32_t getPixelNoPerPyramid(uint8_t nLevels){
	uint32_t subregionDimension = 3 * ((1 << (nLevels + 2)) - 1);
	uint32_t totalPixels = 0;
	for(uint8_t i = 0; i <= nLevels; i++){
		totalPixels += (subregionDimension * subregionDimension);
		subregionDimension = subregionDimension / 2 + (subregionDimension & 1);
	}
	return totalPixels;
}

int main(){
	Image4 *img4 = getStaticImage4();
	Image3 *img = image4to3(img4);
	AlphaMap map = getAlphaMap(img4);
	destroyImage4(&img4);

	llf(img, 0.35, 0.4, 5, 3, 649 255);

	img4 = image3to4AlphaMap(img, map);
	destroyImage3(&img);
	printStaticImage4(img4);
	destroyImage4(&img4);
}