#include <vector>
#include "yocto_colorgrade.h"

#include <yocto/yocto_color.h>
#include <yocto/yocto_sampling.h>

#include <iostream>

namespace yocto {

	std::vector<std::vector<double>> createFilter(){
		std::vector<std::vector<double>> filter(5, std::vector<double>(5, 0));
		double params[] = {0.05, 0.25, 0.4, 0.25, 0.05};

		for(int i = 0; i < 5; i++){
			for(int j = 0; j < 5; j++){
				filter[i][j] = params[i] * params[j];
			}
		}
		return filter;
	}

	color_image subimage(const color_image& image, int startX, int endX, int startY, int endY){
		int w = endX - startX;
		int h = endY - startY;
		color_image ret = make_image(w, h, false);
		for(int y = 0; y < h; y++){
			int finalY = startY + y;
			for(int x = 0; x < w; x++){
				ret[{x, y}] = image[{startX + x, finalY}];
			}
		}
		return ret;
	}

	color_image convolve(color_image& image, std::vector<std::vector<double>>& kernel) {
		auto img = make_image(image.width, image.height, false);
		int  rows = kernel[0].size();
		int  cols = kernel.size();
		int  xstart = -1 * cols / 2;
		int  ystart = -1 * rows / 2;

		vec3f c;
		for (int j = 0; j < img.height; j++) {
			for (int i = 0; i < img.width; i++) {
				c = {0.0f, 0.0f, 0.0f};
				for (int y = 0; y < rows; y++) {
					int jy = j + ystart + y;
					for (int x = 0; x < cols; x++) {
						int ix = i + xstart + x;
						if (ix >= 0 && ix < img.width && jy >= 0 && jy < img.height) {
							auto kern_elem = kernel[x][y];
							auto px = image[{ix, jy}];

							c.x += px.x * kern_elem;
							c.y += px.y * kern_elem;
							c.z += px.z * kern_elem;
						} else {
							auto kern_elem = kernel[x][y];
							auto px = image[{i, j}];

							c.x += px.x * kern_elem;
							c.y += px.y * kern_elem;
							c.z += px.z * kern_elem;
						}
					}
				}
				img[{i, j}].x = c.x;
				img[{i, j}].y = c.y;
				img[{i, j}].z = c.z;
			}
		}
		return img;
	}
}