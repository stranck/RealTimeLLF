#include "../utils/imageutils.h"
#include "../utils/llfUtils.h"
#include "../utils/structs.h"
#include "../utils/vects.h"
#include "../utils/utils.h"
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#include "../utils/test/testimage.h"


/*Image4 downsample(Image4 img, int *width, int *height, double filter[]){
	Image4 I = convolve(img,filter);
	int originalW = *width, originalH = *height;
	*width /= 2;
	*height /= 2;
	Image4 ret = make_image(*width, *height, false);
	int y;
	int startingX = originalW & 1;
	int startingY = originalH & 1;
	for(y = startingY; y < originalH; y += 2) {
		int x;
		for(x = startingX; x < originalW; x += 2) {
			ret[{x / 2, y / 2}] = I[{x - startingX, y - startingY}]; //OCIO maybe it is x - originalW & 1 and y - originalH & 1
		}
	}

	return ret;
}

Image4 upsample(Image4 I, double filter[]){
	int smallWidth = I.width, smallHeight = I.height;
	int uppedW = smallWidth << 1;
	int uppedH = smallHeight << 1;
	Image4 upsampled = make_image(uppedW, uppedH, false);
	for(int y = 0; y < smallHeight; y++){
		int yUp = y * 2;
		int yUpLess = yUp++;
		for(int x = 0; x < smallWidth; x++){
			int xUp = x * 2;
			auto pixel = I[{x, y}];
			int xUpLess = xUp++;

			upsampled[{xUpLess, yUpLess}] = pixel;
			upsampled[{xUpLess, yUp}] = pixel;
			upsampled[{xUp, yUpLess}] = pixel;
			upsampled[{xUp, yUp}] = pixel;
		}
	}
	
	return convolve(upsampled, filter);
}*/

int main(){
	Image4 img4 = getStaticImage4();
	Image3 img = image4to3(img4);
	AlphaMap map = getAlphaMap(img4);
	destroyImage4(img4);
	Pixel3 test = *getPixel3(img, img.height / 2, img.width / 2);
	remap(&img, test, 0.35, 0.4, 5);
	img4 = image3to4AlphaMap(img, map);
	//Pixel4 a = {1, 2, 3, 4};
	//Pixel4 b = {10, 20, 30, 40};
	//int x = 2;
	//Pixel4 delta = /*vec4Add(vec4DivC(*/vec4MulC(vec4MulC(a, x, Pixel4), 2, Pixel4)/*, 2, Pixel4), b, Pixel4)*/;
	//printf("AAAAAAAA %f %f %f %f\n", delta.x, delta.y, delta.z, delta.w);
	printStaticImage4(img4);
	destroyImage4(img4);
}