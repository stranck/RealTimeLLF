#pragma once

#include "structs.h"

Image4 * makeImage4(uint32_t width, uint32_t height);
Image4 * makeImage4WithData(uint32_t width, uint32_t height, Pixel4 pixels[]);
Image4 * makeImage4WithDataPtr(uint32_t width, uint32_t height, Pixel4 *pixels);
Image3 * makeImage3(uint32_t width, uint32_t height);
Image3 * makeImage3WithData(uint32_t width, uint32_t height, Pixel3 pixels[]);

void destroyImage3(Image3 **img);
void destroyImage4(Image4 **img);

inline Pixel4 * getPixel4(Image4 *img, uint32_t x, uint32_t y){
	return &img->pixels[y * img->width + x];
}
inline Pixel4 * getPixel4Vec(Image4 *img, Vec2u32 v){
	return &img->pixels[v.y * img->width + v.x];
}
inline Pixel3 * getPixel3(Image3 *img, uint32_t x, uint32_t y){
	return &img->pixels[y * img->width + x];
}
inline Pixel3 * getPixel3Vec(Image3 *img, Vec2u32 v){
	return &img->pixels[v.y * img->width + v.x];
}

inline void setPixel3(Image3 *img, uint32_t x, uint32_t y, Pixel3 *px){
	img->pixels[y * img->width + x] = *px;
}
inline void setPixel3Vec(Image3 *img, Vec2u32 v, Pixel3 *px){
	img->pixels[v.y * img->width + v.x] = *px;
}

AlphaMap getAlphaMap(Image4 *img);
Image3 * image4to3(Image4 *img);
Image4 * image3to4FixedAlpha(Image3 *img, double alpha);
Image4 * image3to4AlphaMap(Image3 *img, AlphaMap alphaMap);

void subimage3(Image3 *dest, Image3 *source, uint32_t startX, uint32_t endX, uint32_t startY, uint32_t endY);

void imgcpy3(Image3 *dest, Image3 *source);

void fillWithColor(Image3 *dest, Pixel3 *color);

void clampImage3(Image3 *img);
