#include "openmp.h"

#include "../utils/test/testimage.h"

/*
.\bin\ycolorgrade --image tests/flower.png --llf --levels 3 --sigma 0.35 --alpha 0.4 --beta 5
Tested on a r9 5900x
Yocto single core: 21213ms 
Yocto multicore: 1906ms
This single core: 5214ms
This multi core: 224ms
This cuda (gtx 1080): 90ms
*/
int main(int argc, char const *argv[]){
	if(argc < 2){
		printff("Usage: %s <number of threads>\n", argv[0]);
		exit(1);
	}
	int nThreads = atoi(argv[1]);

	Image4 *img4 = getStaticImage4(); //Get the static image bundled with the compiled binary
	Image3 *img = image4to3(img4); //Removes the alpha channel
	AlphaMap map = getAlphaMap(img4); //Gets the alpha map from the original image
	destroyImage4(&img4); //Destroys the original image

	const uint8_t nLevels = 3;
	WorkingBuffers workingBuffers;
	initWorkingBuffers(&workingBuffers, img->width, img->height, nLevels, nThreads); //allocates the buffers used by the llf's processing
	llf(img, 0.35, 0.4, 5, 3, nThreads, &workingBuffers); //Applies local laplacian filter
	destroyWorkingBuffers(&workingBuffers, nLevels, nThreads); //destroy the buffers used by the llf's processing

	img4 = image3to4AlphaMap(img, map); //Readd the alpha channel of the original image
	destroyImage3(&img); //Destroy the image used by the llf algorithm
	printStaticImage4(img4); //Prints the raw bytes of the rendered image
	destroyImage4(&img4); //Destroy the rendered image
}