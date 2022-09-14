#pragma once

#include "../utils/extramath.h"
#include "../utils/structs.h"
#include "../utils/utils.h"
#include "realtime-ndi.h"
#include <cstring>

#ifdef ON_WINDOWS
	#include <windows.h>
	DWORD WINAPI gpuProcessingThread_entryPoint(LPVOID lpParameter);
#else
	#include <pthread.h>
	void * gpuProcessingThread_entryPoint(void *param);
#endif

void startGpuProcessingThread(float sigma, float alpha, float beta, uint8_t nLevels, uint32_t nThreads, uint32_t nBlocks);
void handleIncomingFrame(NDIlib_video_frame_v2_t *ndiVideoFrame);
void writeOutputFrame(NDIlib_video_frame_v2_t *ndiVideoFrame);
void destroyProcessingThread();
void initProcessingThread();
void gpuProcessingThread();