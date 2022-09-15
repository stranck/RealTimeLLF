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

	Image4 *img4 = getStaticImage4();
	Image3 *img = image4to3(img4);
	AlphaMap map = getAlphaMap(img4);
	destroyImage4(&img4);

	const uint8_t nLevels = 3;
	WorkingBuffers workingBuffers;
	initWorkingBuffers(&workingBuffers, img->width, img->height, nLevels, nThreads);
	llf(img, 0.35, 0.4, 5, 3, nThreads, &workingBuffers);
	destroyWorkingBuffers(&workingBuffers, nLevels, nThreads);

	img4 = image3to4AlphaMap(img, map);
	destroyImage3(&img);
	printStaticImage4(img4);
	destroyImage4(&img4);
}