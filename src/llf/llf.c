#include "../utils/structs.h"
#include "../utils/imageutils.h"
#include "../utils/vects.h"
#include "../utils/utils.h"
#include <stdint.h>
#include <stdbool.h>

#include "../utils/test/testimage.h"


Image downsample(Image img, int *width, int *height, double filter[] ){
	Image I = convolve(img,filter);
	int originalW = *width, originalH = *height;
	*width /= 2;
	*height /= 2;
	Image ret = make_image(*width, *height, false);
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

Image upsample(Image I, double filter[]){
	int smallWidth = I.width, smallHeight = I.height;
	int uppedW = smallWidth << 1;
	int uppedH = smallHeight << 1;
	Image upsampled = make_image(uppedW, uppedH, false);
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
}

Image remap(Image I, Vec4f g0, double sigma, double alpha, double beta){
	int size = I.width * I.height;
	auto pixels = I.pixels;
	for(int i = 0; i < size; i++){
		Vec4f delta = pixels[i] - g0;
		double mag = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
		if(mag > 1e-10) {
			delta.x /= mag;
			delta.y /= mag;
			delta.z /= mag;
			delta.w /= mag;
		}	//Ci sono altri modi per dividere ogni elemento contemporaneamente ?

		if(mag < sigma){ //Details
			double fraction = mag / sigma;
			double polynomial = pow(fraction, alpha);
			if(alpha < 1){
				double kNoiseLevel = 0.01;
				double blend = smoothstep(kNoiseLevel, 2 * kNoiseLevel, fraction * sigma);
				polynomial = blend * polynomial + (1 - blend) * fraction;
			}
			I.pixels[i] = g0 + delta * sigma * polynomial;	//Creare una funzione apposita ma non posso ora
		} else { //Edges
			I.pixels[i] = g0 + delta * (((mag - sigma) * beta) + sigma);
		}
	}
	return I;
}

Image make_image(int width, int height, bool linear) {
  //return Image{ width, height, linear, Pixel(width * height, vec4f{0, 0, 0, 0})};
	return;
}

Image convolve(Image image, double kernel[]) {
	return;
}

int main(){
	Image img = getStaticImage();
	printStaticImage(img);
	destroyImage(img);
}