#include <stdint.h>
#include <stdio.h>

#include "openmpStructs.h"
#include "openmpUtils.h"
#include "../utils/utils.h"
#include "../utils/structs.h"



void initLevelInfo(CurrentLevelInfo *cli, uint32_t *pyrDimensions, Pyramid gaussPyramid){
	cli -> lev = -1;
	cli -> nextLevelDimension = 0;
	updateLevelInfo(cli, pyrDimensions, gaussPyramid);
}

void updateLevelInfo(CurrentLevelInfo *cli, uint32_t *pyrDimensions, Pyramid gaussPyramid){
	cli -> lev++;
	cli -> currentGaussLevel = gaussPyramid[cli->lev];
	cli -> width = cli->currentGaussLevel->width;
	cli -> prevLevelDimension = cli->nextLevelDimension;
	cli -> nextLevelDimension += pyrDimensions[cli->lev];
	//printff("Switching level to %d.\t Dim: %dx%d\t PrevDim: %d\t nextDim: %d\n", cli->lev, cli->width, cli->currentGaussLevel->height, cli->prevLevelDimension, cli->nextLevelDimension);
}


void imgcpy3_parallel(Image3 *dest, Image3 *source, const uint8_t nThreads){
	dest->width = source->width;
	dest->height = source->height;
	uint32_t dim = dest->width * dest->height;
	//#pragma omp parallel for num_threads(nThreads) schedule(static, 4)
	for(uint32_t i = 0; i < dim; i++){
		dest->pixels[i] = source->pixels[i];
	}
}