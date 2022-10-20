#include "imageutils.h"

/**
 * @brief Allocates an Image4 of the specified dimension
 * 
 * @param width width of the image
 * @param height height of the image
 * @return Image4* pointer to an heap-allocated image4
 */
Image4 * makeImage4(uint32_t width, uint32_t height){
	Pixel4 *img = (Pixel4 *) malloc(width * height * sizeof(Pixel4));
	Image4 *i = (Image4 *) malloc(sizeof(Image4));
	i -> width = width;
	i -> height = height;
	i -> pixels = img;
	return i;
}
/**
 * @brief Allocates an Image4 of the specified dimension and copies the specified stack-allocated-data over it
 * 
 * @param width width of the image
 * @param height height of the image
 * @param pixels stack-allocated-data to copy over the allocated image
 * @return Image4* pointer to an heap-allocated image4
 */
Image4 * makeImage4WithData(uint32_t width, uint32_t height, Pixel4 pixels[]){
	size_t dimension = width * height * sizeof(Pixel4);
	Pixel4 *img = (Pixel4 *) malloc(dimension);
	memcpy(img, pixels, dimension);
	Image4 *i = (Image4 *) malloc(sizeof(Image4));
	i -> width = width;
	i -> height = height;
	i -> pixels = img;
	return i;
}
/**
 * @brief Allocates an Image4 of the specified dimension and copies the specified heap-allocated-data over it
 * 
 * @param width width of the image
 * @param height height of the image
 * @param pixels heap-allocated-data to copy over the allocated image
 * @return Image4* pointer to an heap-allocated image4
 */
Image4 * makeImage4WithDataPtr(uint32_t width, uint32_t height, Pixel4 *pixels){
	size_t dimension = width * height * sizeof(Pixel4);
	Image4 *i = (Image4 *) malloc(sizeof(Image4));
	i -> width = width;
	i -> height = height;
	i -> pixels = pixels;
	return i;
}
/**
 * @brief Allocates an Image3 of the specified dimension
 * 
 * @param width width of the image
 * @param height height of the image
 * @return Image3* pointer to an heap-allocated image3
 */
Image3 * makeImage3(uint32_t width, uint32_t height){
	Pixel3 *img = (Pixel3 *) malloc(width * height * sizeof(Pixel3));
	Image3 *i = (Image3 *) malloc(sizeof(Image3));
	i -> width = width;
	i -> height = height;
	i -> pixels = img;
	return i;
}
/**
 * @brief Allocates an Image3 of the specified dimension and copies the specified stack-allocated-data over it
 * 
 * @param width width of the image
 * @param height height of the image
 * @param pixels stack-allocated-data to copy over the allocated image
 * @return Image3* pointer to an heap-allocated image3
 */
Image3 * makeImage3WithData(uint32_t width, uint32_t height, Pixel3 pixels[]){
	size_t dimension = width * height * sizeof(Pixel3);
	Pixel3 *img = (Pixel3 *) malloc(dimension);
	memcpy(img, pixels, dimension);
	Image3 *i = (Image3 *) malloc(sizeof(Image3));
	i -> width = width;
	i -> height = height;
	i -> pixels = img;
	return i;
}

/**
 * @brief Deallocates an Image3
 * It also sets its pointer to NULL, to prevent UAF
 * 
 * @param img pointer to the variable holding the image3 to deallocate
 */
void destroyImage3(Image3 **img){
	Image3 *localImg = *img;
	free(localImg -> pixels);
	localImg -> pixels = NULL;
	free(localImg);
	*img = NULL;
}
/**
 * @brief Deallocates an Image4
 * It also sets its pointer to NULL, to prevent UAF
 * 
 * @param img pointer to the variable holding the image4 to deallocate
 */
void destroyImage4(Image4 **img){
	Image4 *localImg = *img;
	free(localImg -> pixels);
	localImg -> pixels = NULL;
	free(localImg);
	*img = NULL;
}

/**
 * @brief Obtains the alpha map from an image4
 * 
 * @param img Image4 used as source
 * @return AlphaMap newly allocated alphamap of the source Image4
 */
AlphaMap getAlphaMap(Image4 *img){
	uint32_t dimension = img->width * img->height;
	AlphaMap map = (AlphaMap) malloc(dimension * sizeof(uint8_t));
	for(uint32_t i = 0; i < dimension; i++)
		map[i] = img->pixels[i].w;
	return map;
}
/**
 * @brief Allocates and returns a new Image3 that's a copy of the source image without the alpha channel
 * 
 * @param img source Image4
 * @return Image3* newly allocated image3 that's a copy of the source except the alpha channel
 */
Image3 * image4to3(Image4 *img){
	Image3 *ret = makeImage3(img -> width, img -> height);
	uint32_t dimension = img->width * img->height;
	for(uint32_t i = 0; i < dimension; i++){
		ret->pixels[i].x = img->pixels[i].x;
		ret->pixels[i].y = img->pixels[i].y;
		ret->pixels[i].z = img->pixels[i].z;
	}
	return ret;
}
/**
 * @brief Allocates and returns a new Image4 that's the copy of the source Image3 with a predefined alpha value
 * 
 * @param img Image without alpha channel
 * @param alpha Predefined alpha value
 * @return Image4* newly allocated copy of the source Image using a predefined alpha value
 */
Image4 * image3to4FixedAlpha(Image3 *img, float alpha){
	Image4 *ret = makeImage4(img -> width, img -> height);
	uint32_t dimension = img->width * img->height;
	for(uint32_t i = 0; i < dimension; i++){
		ret->pixels[i].x = img->pixels[i].x;
		ret->pixels[i].y = img->pixels[i].y;
		ret->pixels[i].z = img->pixels[i].z;
		ret->pixels[i].w = alpha;
	}
	return ret;
}
/**
 * @brief Merges together an Image3 and an alpha map to create an Image4
 * 
 * @param img source Image3
 * @param alphaMap source alphamap
 * @return Image4* newly allocated Image4 that merges both sources
 */
Image4 * image3to4AlphaMap(Image3 *img, AlphaMap alphaMap){
	Image4 *ret = makeImage4(img -> width, img -> height);
	uint32_t dimension = img->width * img->height;
	for(uint32_t i = 0; i < dimension; i++){
		ret->pixels[i].x = img->pixels[i].x;
		ret->pixels[i].y = img->pixels[i].y;
		ret->pixels[i].z = img->pixels[i].z;
		ret->pixels[i].w = alphaMap[i];
	}
	return ret;
}

/**
 * @brief Copy an image3 onto another one
 * 
 * It doesn't check for image size!
 * 
 * @param dest dest image
 * @param source source image
 */
void imgcpy3(Image3 *dest, Image3 *source){
	dest->width = source->width;
	dest->height = source->height;
	memcpy(dest->pixels, source->pixels, dest->width * dest->height * sizeof(Pixel3));
}

/**
 * @brief Cuts a subregion from the source image and copies it onto the dest image
 * 
 * It doesn't check for sizes!
 * 
 * @param dest dest image
 * @param source source image
 * @param startX x base value of the subregion in the source image
 * @param endX x end value of the subregion in the source image
 * @param startY y base value of the subregion in the source image
 * @param endY y end value of the subregion in the source image
 */
void subimage3(Image3 *dest, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY){
	uint32_t w = endX - startX;
	uint32_t h = endY - startY;
	dest->width = w;
	dest->height = h;

	for(uint32_t y = 0; y < h; y++){
		uint32_t finalY = startY + y;
		for(uint32_t x = 0; x < w; x++){
			setPixel3(dest, x, y, getPixel3(source, startX + x, finalY));
		}
	}
}

/**
 * @brief Completely erases an image filling it with the specified color
 * 
 * @param dest Dest image
 * @param color Color used for fill the dest image
 */
void fillWithColor(Image3 *dest, Pixel3 *color){
	uint32_t dim = dest->width * dest->height;
	Pixel3 *pxs = dest->pixels;
	for(uint32_t i = 0; i < dim; i++)
		pxs[i] = *color;
}

/**
 * @brief Clamps an image3 to have each single pixel in the [0;1] boundaries
 * 
 * @param img Image to clamp
 */
void clampImage3(Image3 *img){
	uint32_t dim = img->width * img->height;
	Pixel3 *px = img->pixels;
	for(uint32_t i = 0; i < dim; i++){
		px[i].x = clamp(px[i].x, 0, 1);
		px[i].y = clamp(px[i].y, 0, 1);
		px[i].z = clamp(px[i].z, 0, 1);
	}
}