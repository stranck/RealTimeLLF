#include "llf.h"

#include "../utils/test/testimage.h"

int main(){
	Image4 *img4 = getStaticImage4();
	Image3 *img = image4to3(img4);
	AlphaMap map = getAlphaMap(img4);
	destroyImage4(&img4);

	const uint8_t nLevels = 3;
	WorkingBuffers workingBuffers;
	initWorkingBuffers(&workingBuffers, img->width, img->height, nLevels);
	llf(img, 0.35, 0.4, 5, nLevels, &workingBuffers);
	destroyWorkingBuffers(&workingBuffers, nLevels);

	img4 = image3to4AlphaMap(img, map);
	destroyImage3(&img);
	printStaticImage4(img4);
	destroyImage4(&img4);
}