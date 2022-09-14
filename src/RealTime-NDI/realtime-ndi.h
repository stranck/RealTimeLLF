#pragma once

#include <cstddef>
#include <stdint.h>
#include <stdbool.h>
#include <Processing.NDI.Lib.h>
#include "../utils/utils.h"
#include "processingHandler.h"

//#define CUDA_VERSION 0
//#define OPENMP_VERSION 0

#ifdef ON_WINDOWS
	#ifdef _WIN64
		#pragma comment(lib, "Processing.NDI.Lib.x64.lib")
	#else
		#pragma comment(lib, "Processing.NDI.Lib.x86.lib")
	#endif
#endif

void cleanup();
int main(int argc, char const *argv[]);

#define checkShutdown(){if(shutdownRequested) cleanup();}