#include "llfUtils.h"

/**
 * @brief Remap function to be applied to the subregion inside the llf algorithm
 * It's defined inside the llf's papers https://people.csail.mit.edu/sparis/publi/2011/siggraph/Paris_11_Local_Laplacian_Filters.pdf
 * at page 5 paragaph 4
 * 
 * @param img Image to be remapped
 * @param g0 Reference pixel
 * @param sigma Treshold used by remap function to identify edges and details
 * @param alpha Controls the details level
 * @param beta Controls the tone mapping level
 */
void remap(Image3 * img, const Pixel3 g0, float sigma, float alpha, float beta){
	uint32_t size = img -> width * img -> height;
	Pixel3 *pixels = img -> pixels;
	for(int i = 0; i < size; i++){ //For each pixel of the image
		Pixel3 delta;
		vec3Sub(delta, pixels[i], g0);
		float mag = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
		if(mag > 1e-10)
			vec3DivC(delta, delta, mag);

		if(mag < sigma){ //Details
			float fraction = mag / sigma;
			float polynomial = pow(fraction, alpha);
			if(alpha < 1){
				const float kNoiseLevel = 0.01;
				float blend = smoothstep(kNoiseLevel, 2 * kNoiseLevel, fraction * sigma);
				polynomial = blend * polynomial + (1 - blend) * fraction;
			}
			float d = sigma * polynomial;
			vec3MulC(delta, delta, d);
			vec3Add(img -> pixels[i], g0, delta);
		} else { //Edges
			float d = ((mag - sigma) * beta) + sigma;
			vec3MulC(delta, delta, d);
			vec3Add(img -> pixels[i], g0, delta);
		}
	}
}

/**
 * @brief Allocates a blur kernel for the convolve functions
 * 
 * @return Kernel blur kernel
 */
Kernel createFilter(){
	const float params[KERNEL_DIMENSION] = {0.05, 0.25, 0.4, 0.25, 0.05};
	Kernel filter = (Kernel) malloc(KERNEL_DIMENSION * KERNEL_DIMENSION * sizeof(float));

	for(uint8_t i = 0; i < KERNEL_DIMENSION; i++){
		for(uint8_t j = 0; j < KERNEL_DIMENSION; j++){ //For each kernel element
			filter[getKernelPosition(i, j)] = params[i] * params[j]; //calc it based on the params
		}
	}
	return filter;
}
/**
 * @brief Deallocates a kernel
 * It also sets its pointer to NULL to prevent UAF
 * 
 * @param filter pointer to the variable keeping the kernel
 */
void destroyFilter(Kernel *filter){
	free(*filter);
	filter = NULL;
}

/**
 * @brief Allocates a pyramid with N+1 layers, and the layer 0 of the specified dimensions 
 * 
 * @param width width of the layer 0
 * @param height height of the layer 0
 * @param nLevels number of levels in the pyramid
 * @return Pyramid newly heap-allocated pyramid
 */
Pyramid createPyramid(uint32_t width, uint32_t height, uint8_t nLevels){
	nLevels++; //Pyramids has one more layer!
	Pyramid p = (Pyramid) malloc(nLevels * sizeof(Image3*)); //alloc the pyramid array
	for(uint8_t i = 0; i < nLevels; i++){
		p[i] = makeImage3(width, height); //alloc the image for each layer
		width = width / 2 + (width & 1);
		height = height / 2 + (height & 1); //update the dimensions of the next layer
	}
	return p;
}
/**
 * @brief De-allocates an already existing pyramid and its layers
 * It also set the pointer to the pyramid to NULL to prevent UAF
 * 
 * @param p Pointer to the variable keeping the already allocated pyramid
 * @param nLevels number of layers of the pyramid
 */
void destroyPyramid(Pyramid *p, uint8_t nLevels){
	Pyramid p_local = *p;
	for(uint8_t i = 0; i <= nLevels; i++){
		destroyImage3(&p_local[i]); //destroy every layer
	}
	free(p_local); //free the pyramid array
	p = NULL;
}

/**
 * @brief Applies a kernel over an image
 * 
 * @param dest destination image
 * @param source source image
 * @param kernel kernel to apply
 */
void convolve(Image3 *dest, Image3 *source, Kernel kernel) {
	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;

	for (uint32_t j = 0; j < dest->height; j++) {
		for (uint32_t i = 0; i < dest->width; i++) { //For each pixel C of the image
			Pixel3 c = zero3vect;
			for (uint32_t y = 0; y < rows; y++) {
                int32_t jy = j + ystart + y;
				for (uint32_t x = 0; x < cols; x++) { //For each pixel of the kernel square surrounding C 
                    int32_t ix = i + xstart + x;
                    if (ix >= 0 && ix < dest->width && jy >= 0 && jy < dest->height) { //If we're in bounds
						float kern_elem = kernel[getKernelPosition(x, y)]; //Take the kernel element
						Pixel3 px = *getPixel3(source, ix, jy); //Take the pixel inside the kernel square

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem; //Apply the kernel element to C
					} else {
						float kern_elem = kernel[getKernelPosition(x, y)];
						Pixel3 px = *getPixel3(source, i, j); //If we're out of bounds we will take or 0, or the image limit

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					}
				}
			}
			setPixel3(dest, i, j, &c); //Apply the blurred pixel C to the dest image
		}
	}
}

/**
 * @brief Upsamples an image by duplicating it in size and the applying a blur kernel to remove the squares at the same time
 * 
 * This will save an extra copy of the whole upsized image and the need of an extra temp buffer
 * 
 * @param dest Destination image
 * @param source Source image 
 * @param kernel Blur kernel
 */
void upsampleConvolve(Image3 *dest, Image3 *source, Kernel kernel){
	const uint32_t smallWidth = source->width, smallHeight = source->height;
	const uint32_t uppedW = smallWidth << 1;
	const uint32_t uppedH = smallHeight << 1;
	dest->width = uppedW;
	dest->height = uppedH; //Duplicate the size of source and save it in dest
	const uint8_t  rows = KERNEL_DIMENSION;
	const uint8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;

	for (uint32_t j = 0; j < uppedH; j++) {
		for (uint32_t i = 0; i < uppedW; i++) { //For each pixel C of the upsized image
			Pixel3 c = zero3vect;
			for (uint32_t y = 0; y < rows; y++) { //For each pixel in a KERNEL_DIMENSION^2 square sorrounding C
                int32_t jy = (j + ystart + y) / 2;
				for (uint32_t x = 0; x < cols; x++) {
                    int32_t ix = (i + xstart + x) / 2; //Half the coordinate to use them on the original smaller image
                    if (ix >= 0 && ix < smallWidth && jy >= 0 && jy < smallHeight) { //If we're in bounds of the smaller image
						float kern_elem = kernel[getKernelPosition(x, y)]; //Take the kernel element
						Pixel3 px = *getPixel3(source, ix, jy); //Take the pixel from the smaller image

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem; //Apply the kernel element
					} else {
						float kern_elem = kernel[getKernelPosition(x, y)];
						Pixel3 px = *getPixel3(source, i / 2, j / 2);

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem; //If we're out of bounds we will take or 0, or the image limit
					}
				}
			}
			setPixel3(dest, i, j, &c); //Apply the blurred pixel C to the upsized image
		}
	}
}
/**
 * @brief Creates a laplacian pyramid starting from a gauss pyramid
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
void laplacianPyramid(Pyramid laplacian, Pyramid tempGauss, uint8_t nLevels, Kernel filter){
	for(uint8_t i = 0; i < nLevels; i++){
		Image3 *upsampled = laplacian[i];
		upsampleConvolve(upsampled, tempGauss[i + 1], filter); //Upsample the next layer using laplacian[i] as temp buffer

		Image3 *current = tempGauss[i];
		uint32_t yEnd = llf_min(current->height, upsampled->height);
		uint32_t xEnd = llf_min(current->width, upsampled->width);
		for (uint32_t y = 0; y < yEnd; y++){
			for (uint32_t x = 0; x < xEnd; x++){ //For each pixel
				Pixel3 *upsPtr = getPixel3(upsampled, x, y);
				Pixel3 ups = *upsPtr;
				Pixel3 crr = *getPixel3(current, x, y); //Get the pixel from both upsampled and current gauss layers 

				vec3Sub(*upsPtr, crr, ups); //subtract and store them inside the current laplacian pyramid layer
			}
		}
	}
	imgcpy3(laplacian[nLevels], tempGauss[nLevels]); //Manually copy the smallest layer from the gauss to the laplacian pyramid
}


/**
 * @brief Downsamples an image by halfing it in size and the applying a blur kernel to remove the gaps at the same time
 * 
 * This will save an extra copy of the whole downsized image and the need of an extra temp buffer
 * 
 * @param dest destination bigger image
 * @param source source smaller image
 * @param width pointer to the width of the source image
 * @param height pointer to the height of the source image
 * @param filter blur kernel
 */
void downsampleConvolve(Image3 *dest, Image3 *source, uint32_t *width, uint32_t *height, Kernel filter){
	const uint32_t originalW = *width, originalH = *height;
	*width /= 2;
	*height /= 2;
	dest->width = *width;
	dest->height = *height; //Half the image dimension and save both of them in the original ptrs and inside the dest image
	const int32_t startingX = originalW & 1;
	const int32_t startingY = originalH & 1; //If the dimension is odd, we copy only the "middle" pixels. Eg the X: -X-X-
	const int8_t  rows = KERNEL_DIMENSION;
	const int8_t  cols = KERNEL_DIMENSION;
	const int32_t  xstart = -1 * cols / 2;
	const int32_t  ystart = -1 * rows / 2;

	for (uint32_t j = startingY; j < originalH; j += 2) {
		for (uint32_t i = startingX; i < originalW; i += 2) { //For half of the pixels C
			Pixel3 c = zero3vect;
			for (uint32_t y = 0; y < rows; y++) {
				int32_t jy = j + (ystart + y) * 2 - startingY;
				for (uint32_t x = 0; x < cols; x++) { //For each pixel in a KERNEL_DIMENSION^2 square sorrounding C
					int32_t ix = i + (xstart + x) * 2 - startingX; //Double the coordinate to use them on the original bigger image

					if (ix >= 0 && ix < originalW && jy >= 0 && jy < originalH) { //If we're in bounds of the bigger image
						float kern_elem = filter[getKernelPosition(x, y)]; 
						Pixel3 px = *getPixel3(source, ix, jy); //Take the pixel from the bigger image

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem; //Apply the kernel element
					} else {
						
						float kern_elem = filter[getKernelPosition(x, y)];
						Pixel3 px = *getPixel3(source, i - startingX, j - startingY); //If we're out of bounds we will take or 0, or the image limit

						c.x += px.x * kern_elem;
						c.y += px.y * kern_elem;
						c.z += px.z * kern_elem;
					}
				}
			}
			setPixel3(dest, i / 2, j / 2, &c); //Apply the blurred pixel C to the downsized image
		}
	}
}
/**
 * @brief Creates a gaussian pyramid starting from a source image
 * 
 * Each single layer of a gaussian pyramid is defined as follows:
 * gauss[0] = sourceImg
 * gauss[n] = downsample(gauss[n - 1])
 * 
 * @param outPyr output gaussian pyramid
 * @param inImg source image
 * @param nLevels number of layers of the pyramid
 * @param filter blur kernel
 */
void gaussianPyramid(Pyramid outPyr, Image3 *inImg, uint8_t nLevels, Kernel filter){
	imgcpy3(outPyr[0], inImg); //Copy the first layer
	uint32_t width = inImg->width, height = inImg->height;
	//if(0 <= nLevels){ //So it don't need to copy two times the whole img
		downsampleConvolve(outPyr[1], inImg, &width, &height, filter);
	//}
	for(uint8_t i = 1; i < nLevels; i++)
		downsampleConvolve(outPyr[i + 1], outPyr[i], &width, &height, filter); //Downsample the current layer and save it into the next one
}