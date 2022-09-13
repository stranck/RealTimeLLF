#include "cuda.cuh"

#include "../utils/test/testimage.h"

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

	const uint8_t nLevels = 2;
	CUDAbuffers cudaBuffers;
	initCUDAbuffers(&cudaBuffers, img->width, img->height, nLevels);
	llf(img, 0.35, 0.4, 5, nLevels, threadsNo, blocksNo, &cudaBuffers);
	destroyCUDAbuffers(&cudaBuffers, nLevels);

	img4 = image3to4AlphaMap(img, map);
	destroyImage3(&img);
	printStaticImage4(img4);
	destroyImage4(&img4);
}