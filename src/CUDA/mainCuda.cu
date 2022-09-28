#include "cuda.cuh"

#include "../utils/test/testimage.h"

int main(int argc, char const *argv[]){
	if(argc < 3){
		printff("Usage: %s <number of blocks> <number of threads>\n", argv[0]);
		exit(1);
	}
	int blocksNo = atoi(argv[1]);
	int threadsNo = atoi(argv[2]);
	
	Image4 *img4 = getStaticImage4(); //Get the static image bundled with the compiled binary
	Image3 *img = image4to3(img4); //Removes the alpha channel
	AlphaMap map = getAlphaMap(img4); //Gets the alpha map from the original image
	destroyImage4(&img4); //Destroys the original image

	const uint8_t nLevels = 2;
	WorkingBuffers cudaBuffers;
	initWorkingBuffers(&cudaBuffers, img->width, img->height, nLevels); //allocates the buffers used by the llf's processing
	llf(img, 0.35, 0.4, 5, nLevels, threadsNo, blocksNo, &cudaBuffers); //Applies local laplacian filter
	destroyWorkingBuffers(&cudaBuffers, nLevels); //destroy the buffers used by the llf's processing

	img4 = image3to4AlphaMap(img, map); //Readd the alpha channel of the original image
	destroyImage3(&img); //Destroy the image used by the llf algorithm
	printStaticImage4(img4); //Prints the raw bytes of the rendered image
	destroyImage4(&img4); //Destroy the rendered image
}