#pragma once

#include "structs.h"

Image makeImage(uint32_t width, uint32_t height);
Image makeImageWithData(uint32_t width, uint32_t height, Pixel pixels[]);

inline Pixel * getPixel(Image img, uint32_t x, uint32_t y){
	return &img.pixels[y * img.width + x];
}
inline Pixel * getPixelVec(Image img, Vec2u32 v){
	return &img.pixels[v.y * img.width + v.x];
}

void destroyImage(Image img);