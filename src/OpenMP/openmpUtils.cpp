#include <stdint.h>
#include <stdio.h>

#include "openmpStructs.h"
#include "openmpUtils.h"
#include "../utils/utils.h"
#include "../utils/structs.h"
#include "../utils/extramath.h"


/**
 * @brief Initialize the object used by each thread to have info about the current gaussian level they're working on
 * It does that by first initializing the level to -1, and then by calling the update function to load level 0
 * 
 * @param cli Current level info object 
 * @param pyrDimensions Cached number of pixel per each layer of the gaussian pyramid
 * @param gaussPyramid Original gaussian pyramid
 */
void initLevelInfo(CurrentLevelInfo *cli, uint32_t *pyrDimensions, Pyramid gaussPyramid){
	cli -> lev = -1; 
	cli -> nextLevelDimension = 0;
	cli -> oldY = 0xfffffff;
	updateLevelInfo(cli, pyrDimensions, gaussPyramid);
}

/**
 * @brief Updates the status of the info object to the next level of the gaussian pyramid
 * 
 * @param cli Current level info object
 * @param pyrDimensions Cached number of pixel per each layer of the gaussian pyramid
 * @param gaussPyramid Original gaussian pyramid
 */
void updateLevelInfo(CurrentLevelInfo *cli, uint32_t *pyrDimensions, Pyramid gaussPyramid){
	cli -> lev++; //We go to the next layer of the gaussian pyramid
	//Update cached infos
	cli -> currentNLevels = cli->lev + 1;
	cli -> subregionDimension = 3 * ((1 << (cli->lev + 2)) - 1) / 2;
	cli -> currentGaussLevel = gaussPyramid[cli->lev];
	cli -> width = cli->currentGaussLevel->width;
	cli -> prevLevelDimension = cli->nextLevelDimension; //Updates the number of pixels we worked on the previous frames
	cli -> nextLevelDimension += pyrDimensions[cli->lev]; //Updates the total number of pixels that will be rendered at the end of this layer. This is used to check when we're done with it
}

/**
 * @brief Copy an image3 onto another one using multiple threads
 * 
 * It doesn't check for image size!
 * 
 * @param dest dest image
 * @param source source image
 * @param nThreads number of threads used in the copy operation
 */
void imgcpy3_parallel(Image3 *dest, Image3 *source, const uint8_t nThreads){
	dest->width = source->width;
	dest->height = source->height;
	uint32_t dim = dest->width * dest->height;

	#pragma omp parallel for num_threads(nThreads) schedule(static, 8)
	for(int32_t i = 0; i < dim; i++){
		dest->pixels[i] = source->pixels[i];
	}
}

/**
 * @brief Clamps an image3 to have each single pixel in the [0;1] boundaries using multiple threads
 * 
 * @param img Image to clamp
 * @param nThreads Number of threads that will concurrently clamp the image
 */
void clampImage3_parallel(Image3 *img, const uint8_t nThreads){
	int32_t dim = img->width * img->height;
	Pixel3 *px = img->pixels;
	#pragma omp parallel for num_threads(nThreads) schedule(static, 8)
	for(int32_t i = 0; i < dim; i++){
		px[i].x = clamp(px[i].x, 0, 1);
		px[i].y = clamp(px[i].y, 0, 1);
		px[i].z = clamp(px[i].z, 0, 1);
	}
}