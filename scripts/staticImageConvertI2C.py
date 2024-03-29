import sys
from PIL import Image

FILE = """
#pragma once

#include "../structs.h"
#include "../imageutils.h"
#include "../utils.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

const uint32_t DATA[] = {%DATA%
};

Image4 * getStaticImage4(){
	const uint32_t width = %WIDTH%, height = %HEIGHT%;
	const uint32_t dim = width * height;

	Pixel4 *pxs = (Pixel4 *) malloc(width * height * sizeof(Pixel4));
	for(uint32_t i = 0; i < dim; i++){
		// 1 : out = 255 : in
		uint8_t r = (DATA[i] >> 24) & 0xff;
		uint8_t g = (DATA[i] >> 16) & 0xff;
		uint8_t b = (DATA[i] >> 8) & 0xff;
		uint8_t a = DATA[i] & 0xff;
		Pixel4 p = {r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f};
		pxs[i] = p;
	}
	Image4 *img = makeImage4WithDataPtr(width, height, pxs);
	return img;
}

void printStaticImage4(Image4 *img){
	Pixel4 *pxs = img->pixels;
	const uint32_t width = img->width;
	const uint32_t height = img->height;
	printf("STATIC_IMG_DIMENSIONS: %d %d\\n", width, height);
	Pixel4u8 *buffer = (Pixel4u8 *) calloc(width, sizeof(Pixel4u8));
	for(uint32_t i = 0; i < height; i++){
		Pixel4 *currentLine = &pxs[width * i];
		for(uint32_t j = 0; j < width; j++){
			buffer[j].x = roundfu8(255.0f * currentLine[j].x);
			buffer[j].y = roundfu8(255.0f * currentLine[j].y);
			buffer[j].z = roundfu8(255.0f * currentLine[j].z);
			buffer[j].w = roundfu8(255.0f * currentLine[j].w);
		}
		printBuffer((uint8_t *) buffer, width * sizeof(Pixel4u8));
		puts("");
	}
}
"""
def getPixel(px):
	n = px[0]
	n = n << 8 | px[1]
	n = n << 8 | px[2]
	n = n << 8 | px[3]
	return f"{hex(n)}, "
CHANGE_ROW = "\n\t\t"


args = sys.argv
if(len(args) < 3):
	print(f"{args[0]} <input image> <output .h file>")
	exit(1)
im = Image.open(args[1])
rgba = im.convert('RGBA')

out = ""

for y in range(rgba.height):
	out += CHANGE_ROW
	for x in range(rgba.width):
		out += getPixel(rgba.getpixel((x, y)))
out = out[:-2]

print(f"Original dimensions: {rgba.width}x{rgba.height}")
fileOut = FILE.replace("%WIDTH%", f"{rgba.width}").replace("%HEIGHT%", f"{rgba.height}").replace("%DATA%", out)
f = open(args[2], "w")
f.write(fileOut)
f.close()