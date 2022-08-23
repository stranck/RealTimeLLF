#include "cuda.cuh"

__device__ void upsampleConvolve(Image3 *dest, Image3 *source, Kernel kernel){
	__shared__ Pixel3 ds_upsampled[MAX_PYR_LAYER * MAX_PYR_LAYER];
	
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
	
	uint32_t dim = smallWidth * smallHeight;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;
		if(idx < dim){
			uint32_t x = idx % smallWidth, y = idx / smallWidth;
			ds_upsampled[y * smallWidth + x] = d_getPixel3(source, x, y);
		}
	}
	__syncthreads();

	dim = uppedW * uppedH;
	max = dim / blockDim.x;
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
					Pixel3 px = ds_upsampled[fj * uppedW + fi];
					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
				}
			}
			d_setPixel3(dest, i, j, c);
		}
	}
	__syncthreads();
}

__device__ void downsampleConvolve(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter){
	__shared__ Pixel3 ds_downsampled[MAX_PYR_LAYER * MAX_PYR_LAYER];
	const uint32_t originalW = *width, originalH = *height;
	uint32_t lcl_width = *width / 2;
	if(threadIdx.x == 0){
		*width = lcl_width;
		*height /= 2;
		dest->width = lcl_width;
		dest->height = *height;
	}
	__syncthreads();
	uint32_t startingX = originalW & 1;
	uint32_t startingY = originalH & 1;
	
	uint32_t dim = lcl_width * *height;
	uint32_t max = dim / blockDim.x;
	for(uint32_t i = 0; i <= max; i++){
		uint32_t idx = i * blockDim.x + threadIdx.x;

		if(idx < dim){
			uint32_t x = idx % lcl_width, y = idx / lcl_width;
			ds_downsampled[y * lcl_width + x] = d_getPixel3(source, (x * 2) - startingX, (y * 2) - startingY);
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
			uint32_t i = idx % lcl_width, j = idx / lcl_width;
			Pixel3 c = zero3vect;
			for (int32_t y = 0; y < rows; y++) {
				int32_t jy = j + ystart + y;
				for (int32_t x = 0; x < cols; x++) {
					int32_t ix = i + xstart + x;

					int32_t oob = ix >= 0 && ix < dest->width && jy >= 0 && jy < dest->height;
					int32_t fi = ix * oob + i * (1 - oob), fj = jy * oob + j * (1 - oob);

					double kern_elem = filter[getKernelPosition(x, y)];
					Pixel3 px = ds_downsampled[fj * lcl_width + fi];
					c.x += px.x * kern_elem;
					c.y += px.y * kern_elem;
					c.z += px.z * kern_elem;
				}
			}
			d_setPixel3(dest, i, j, c);
		}
	}
	__syncthreads();
}

__global__ void gaussianPyramid(Pyramid d_outPyr, Image3 *d_inImg, uint8_t nLevels, Kernel d_filter){
	__gaussianPyramid_internal(d_outPyr, d_inImg, nLevels, d_filter);
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

__device__ void laplacianPyramid(Pyramid laplacian, Pyramid tempGauss, uint8_t nLevels, Kernel filter){
	for(uint8_t i = 0; i < nLevels; i++){
		Image3 *upsampled = laplacian[i];
		upsampleConvolve(upsampled, tempGauss[i + 1], filter);
		//No extra synchtreads needed because there already is one at the end of upsampleConvolve 

		Image3 *current = tempGauss[i];
		//TODO Check if min macro works fine for cuda
		uint32_t yEnd = min(current->height, upsampled->height);
		uint32_t xEnd = min(current->width, upsampled->width);
		uint32_t dim = xEnd * yEnd;
		uint32_t max = dim / blockDim.x;
		for(uint32_t li = 0; li <= max; li++){
			uint32_t idx = li * blockDim.x + threadIdx.x;
			if(idx < dim){
				uint32_t x = idx % xEnd, y = idx / xEnd;

				Pixel3 ups = d_getPixel3(upsampled, x, y);
				Pixel3 crr = d_getPixel3(current, x, y);

				d_setPixel3(upsampled, x, y, vec3Sub(crr, ups, Pixel3));
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
			biggerLevelPxs[px] = vec3Add(destPxs[px], biggerLevelPxs[px], Pixel3);
	}
	__syncthreads();
}

__global__ void __d_llf_internal(Pyramid outputLaplacian, Pyramid gaussPyramid, Image3 *img, uint32_t width, uint32_t height, uint8_t lev, uint32_t subregionDimension, Kernel filter, double sigma, double alpha, double beta, PyrBuffer *buffer){
	__shared__ double lcl_filter[KERNEL_DIMENSION * KERNEL_DIMENSION];
	Image3 *currentGaussLevel = gaussPyramid[lev];
	uint32_t x = blockIdx.x, y = blockIdx.y;

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
	NodeBuffer *node;
	if(threadIdx.x == 0){
		node = d_aquireBuffer(buffer);
		bufferLaplacianPyramid = node->bufferLaplacianPyramid;
		bufferGaussPyramid = node->bufferGaussPyramid;

		g0 = d_getPixel3(currentGaussLevel, x, y);
	}
	__syncthreads();

	d_subimage3(bufferLaplacianPyramid[0], img, base_x, end_x, base_y, end_y); //Using bufferLaplacianPyramid[0] as temp buffer
	d_remap(bufferLaplacianPyramid[0], g0, sigma, alpha, beta);
	uint8_t currentNLevels = lev + 1;
	__gaussianPyramid_internal(bufferGaussPyramid, bufferLaplacianPyramid[0], currentNLevels, lcl_filter);
	laplacianPyramid(bufferLaplacianPyramid, bufferGaussPyramid, currentNLevels, lcl_filter);

	if(threadIdx.x == 0){
		d_setPixel3(outputLaplacian[lev], x, y, d_getPixel3(bufferLaplacianPyramid[lev], full_res_roi_x >> lev, full_res_roi_yShifted)); //idk why i had to shift those
		
		d_releaseBuffer(node, buffer);
	}
	__syncthreads();
}

__host__ void llf(Image3 *h_img, double h_sigma, double h_alpha, double h_beta, uint8_t h_nLevels, uint32_t h_nThreads, uint32_t h_elementsNo){
	uint32_t h_width = h_img->width, h_height = h_img->height;
	h_nLevels = min(h_nLevels, 5);
	h_nLevels = max(h_nLevels, 3);//int(ceil(std::abs(std::log2(min(width, height)) - 3))) + 2;
	Kernel d_filter = createFilterDevice();
	Pyramid d_gaussPyramid = createPyramid(h_width, h_height, h_nLevels);
	Pyramid d_outputLaplacian = createPyramid(h_width, h_height, h_nLevels);

	PyrBuffer *d_buffer = createBufferDevice(h_elementsNo, (3 * ((1 << (h_nLevels + 1)) - 1)), h_nLevels);

	Image3 *d_img = makeImage3Device(h_width, h_height);
	copyImg3Host2Device(d_img, h_img);
	gaussianPyramid<<<1, h_nThreads>>>(d_gaussPyramid, d_img, h_nLevels, d_filter);
	CHECK(cudaDeviceSynchronize());

	for(uint8_t h_lev = 0; h_lev < h_nLevels; h_lev++){
		uint32_t h_layerW, h_layerH;
		getPyramidDimensionsAtLayer(d_gaussPyramid, h_lev, &h_layerW, &h_layerH);
		dim3 grid(h_layerW, h_layerH);
		uint32_t h_subregionDimension = 3 * ((1 << (h_lev + 2)) - 1) / 2;

		__d_llf_internal<<<grid, h_nThreads>>>(d_outputLaplacian, d_gaussPyramid, d_img, h_width, h_height, h_lev, h_subregionDimension, d_filter, h_sigma, h_alpha, h_beta, d_buffer);
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

	llf(img, 0.35, 0.4, 5, 3, 128, 1);

	img4 = image3to4AlphaMap(img, map);
	destroyImage3(&img);
	printStaticImage4(img4);
	destroyImage4(&img4);
}