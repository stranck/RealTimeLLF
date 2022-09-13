#pragma once

#include <cstddef>
#include <stdint.h>
#include <stdbool.h>
#include <Processing.NDI.Lib.h>
#include "../utils/utils.h"
#include "cudaHandler.cuh"

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