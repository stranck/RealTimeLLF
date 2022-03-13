#include "../utils/structs.h"
#include "../utils/imageutils.h"
#include "../utils/vects.h"
#include "../utils/utils.h"
#include <stdint.h>

#include "../utils/test/testimage.h"

int main(){
	Image img = getStaticImage();
	printStaticImage(img);
	destroyImage(img);
}