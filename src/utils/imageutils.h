#pragma once

#include "structs.h"

Image4 makeImage4(uint32_t width, uint32_t height);
Image4 makeImage4WithData(uint32_t width, uint32_t height, Pixel4 pixels[]);
Image3 makeImage3(uint32_t width, uint32_t height);
Image3 makeImage3WithData(uint32_t width, uint32_t height, Pixel3 pixels[]);

inline Pixel4 * getPixel4(Image4 img, uint32_t x, uint32_t y){
	return &img.pixels[y * img.width + x];
}
inline Pixel4 * getPixel4Vec(Image4 img, Vec2u32 v){
	return &img.pixels[v.y * img.width + v.x];
}
inline Pixel3 * getPixel3(Image3 img, uint32_t x, uint32_t y){
	return &img.pixels[y * img.width + x];
}
inline Pixel3 * getPixel3Vec(Image3 img, Vec2u32 v){
	return &img.pixels[v.y * img.width + v.x];
}

uint8_t * getAlphaMap(Image4 img);
Image3 image4to3(Image4 img);
Image4 image3to4FixedAlpha(Image3 img, double alpha);
Image4 image3to4AlphaMap(Image3 img, uint8_t *alphaMap);

void destroyImage4(Image4 img);
void destroyImage3(Image3 img);