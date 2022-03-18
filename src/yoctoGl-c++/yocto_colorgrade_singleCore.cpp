//
// Implementation for Yocto/Grade.
//

//
// LICENSE:
//
// Copyright (c) 2020 -- 2020 Fabio Pellacini
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include "yocto_colorgrade.h"

#include <yocto/yocto_color.h>
#include <yocto/yocto_sampling.h>

#include <iostream>

#include "utils.cpp"


// -----------------------------------------------------------------------------
// COLOR GRADING FUNCTIONS
// -----------------------------------------------------------------------------
namespace yocto {
	color_image llf(color_image& I, const grade_params& params);

	color_image grade_image(const color_image& image, const grade_params& params) {
		auto img = image; //No idea why I have to do this and I can't pass image directly
		if(params.llf)
			img = llf(img, params);

		return img;
	}

	color_image downsample(color_image& img, int *width, int *height, std::vector<std::vector<double>>& filter){
		color_image I = convolve(img, filter);
		int originalW = *width, originalH = *height;
		*width /= 2;
		*height /= 2;
		color_image ret = make_image(*width, *height, false);
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
	
	color_image upsample(color_image& I, std::vector<std::vector<double>>& filter){
		int smallWidth = I.width, smallHeight = I.height;
		int uppedW = smallWidth << 1;
		int uppedH = smallHeight << 1;
		color_image upsampled = make_image(uppedW, uppedH, false);
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

	color_image remap(color_image& I, vec4f& g0, double sigma, double alpha, double beta){
		int size = I.width * I.height;
		auto pixels = I.pixels;
		for(int i = 0; i < size; i++){
			vec4f delta = pixels[i] - g0;
			double mag = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
			if(mag > 1e-10) delta /= mag;

			if(mag < sigma){ //Details
				double fraction = mag / sigma;
				double polynomial = pow(fraction, alpha);
				if(alpha < 1){
					double kNoiseLevel = 0.01;
					double blend = smoothstep(kNoiseLevel, 2 * kNoiseLevel, fraction * sigma);
					polynomial = blend * polynomial + (1 - blend) * fraction;
				}
				I.pixels[i] = g0 + delta * sigma * polynomial;
			} else { //Edges
				I.pixels[i] = g0 + delta * (((mag - sigma) * beta) + sigma);
			}
		}
		return I;
	}

	color_image collapse(std::vector<color_image> laplacianPyr, int nLevels, std::vector<std::vector<double>>& filter){
		color_image result = laplacianPyr.back();

		for(int lev = laplacianPyr.size() - 2; lev >= 0; lev--){			
			color_image pyr = laplacianPyr[lev];
			int pyrWidth = pyr.width, pyrHeight = pyr.height;
			result = upsample(result, filter);

			auto psxPyr = pyr.pixels, psxUpsampled = result.pixels;
			int sizeUpsampled = min(result.width, pyrWidth) * min(result.height, pyrHeight);
			for(int px = 0; px < sizeUpsampled; px++)
				result.pixels[px] = psxPyr[px] + psxUpsampled[px];
		}
		std::cout << "Returning from collapse\n";
		return result;
	}
	std::vector<color_image> gaussian_pyramid(color_image& img, int nLevels, std::vector<std::vector<double>>& filter){
		std::vector<color_image> gPyramid;
		gPyramid.push_back(img);

		int width = img.width, height = img.height;
		//if(0 <= nLevels){ //So it don't need to copy two times the whole img
			color_image I = downsample(img, &width, &height, filter);
			gPyramid.push_back(I);
		//}
		for(int i = 1; i < nLevels; i++){
			I = downsample(I, &width, &height, filter);
			gPyramid.push_back(I);
		}

		return gPyramid;
	}

	std::vector<color_image> laplacian_pyramid(std::vector<color_image>& tempGauss, std::vector<std::vector<double>>& filter){
		std::vector<color_image> laplacian;
		int levMax = tempGauss.size() - 1;
		for(int i = 0; i < levMax; i++){ //Not sure about the -1
			color_image upsampled = upsample(tempGauss[i + 1], filter);


			auto current = tempGauss[i];
			auto yEnd = min(current.height, upsampled.height);
			auto xEnd = min(current.width, upsampled.width);
			for (int y = 0; y < yEnd; y++)
				for (int x = 0; x < xEnd; x++)
					upsampled[{x, y}] = current[{x, y}] - upsampled[{x, y}];

			laplacian.push_back(upsampled);
		}
		laplacian.push_back(tempGauss[levMax]);
		return laplacian;
	}

	color_image llf(color_image& I, const grade_params& params) {
		int width = I.width, height = I.height;
		double sigma = params.sigma;
		double alpha = params.alpha;
		double beta = params.beta;
		int nLevels = max(min(params.levelNo, 5), 3);//int(ceil(std::abs(std::log2(min(width, height)) - 3))) + 2;
		std::vector<std::vector<double>> filter = createFilter();

		std::vector<color_image> gaussPyramid = gaussian_pyramid(I, nLevels, filter);
		std::vector<color_image> outputLaplacian;
		for(int i = 0; i < nLevels; i++){
			auto a = gaussPyramid[i];
			outputLaplacian.push_back(make_image(a.width, a.height, false));
		}
		
		for(int lev = 0; lev < nLevels; lev++){
			color_image currentGaussLevel = gaussPyramid[lev];
			int gaussianWidth = currentGaussLevel.width, gaussianHeight = currentGaussLevel.height;
			int subregionDimension = 3 * ((1 << (lev + 2)) - 1) / 2;

			for(int y = 0; y < gaussianHeight; y++){
				std::cout << "laplacian inner loop " << lev << "/" << (nLevels - 1) << "\ty = " << y << "/" << gaussianHeight << "\n";

				//no fuckin clues what this calcs are
				int full_res_y = (1 << lev) * y;
				int roi_y0 = full_res_y - subregionDimension;
				int roi_y1 = full_res_y + subregionDimension + 1;
				int base_y = max(0, roi_y0);
				int end_y = min(roi_y1, height);
				int full_res_roi_y = full_res_y - base_y;
				int full_res_roi_yShifted = full_res_roi_y >> lev;

				for(int x = 0; x < gaussianWidth; x++){
					//no fuckin clues what this calcs are PT2
					int full_res_x = (1 << lev) * x;
					int roi_x0 = full_res_x - subregionDimension;
					int roi_x1 = full_res_x + subregionDimension + 1;
					int base_x = max(0, roi_x0);
					int end_x = min(roi_x1, width);
					int full_res_roi_x = full_res_x - base_x;

					vec4f g0 = currentGaussLevel[{x, y}];
					color_image subregion = subimage(I, base_x, end_x, base_y, end_y);
					color_image remapped = remap(subregion, g0, sigma, alpha, beta);

					std::vector<color_image> tempGauss = gaussian_pyramid(remapped, lev + 1, filter);
					std::vector<color_image> tempLaplacian = laplacian_pyramid(tempGauss, filter);

					outputLaplacian[lev][{x, y}] = tempLaplacian[lev][{full_res_roi_x >> lev, full_res_roi_yShifted}]; //idk why i had to shift those
				}
			}
		}
		outputLaplacian.push_back(gaussPyramid[nLevels]); //gauss pyramid has one layer more
		std::cout << "Collapsing\n";
		return collapse(outputLaplacian, nLevels - 1, filter);
	}
}  // namespace yocto