#include <stdint.h>

#include "openmpStructs.h"
#include "openmpExtra.h"
#include "../structs.h"


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
	cli -> nextLevelDimension += pyrDimensions[cli->lev + 1];
}