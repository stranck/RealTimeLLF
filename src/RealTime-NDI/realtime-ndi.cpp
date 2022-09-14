#include "realtime-ndi.h"

volatile bool shutdownRequested = false;
#ifdef ON_WINDOWS
	#include <windows.h> 

	BOOL WINAPI consoleHandler(DWORD signal) {
		if(signal == CTRL_C_EVENT)
			shutdownRequested = true;
		return true;
	}
#else
	#include <signal.h>
	#include <stdlib.h>

	void sigintHandler(int signal){
		shutdownRequested = true;
	}
#endif

NDIlib_recv_instance_t ndiReceiver;
NDIlib_find_instance_t ndiFinder;
NDIlib_send_instance_t ndiSender;

void cleanup(){
	print("Shutting down NDI");
	if(ndiReceiver) NDIlib_recv_destroy(ndiReceiver);
	if(ndiFinder) NDIlib_find_destroy(ndiFinder);
	if(ndiSender) NDIlib_send_destroy(ndiSender);
	NDIlib_destroy();
	puts("Shuttind down processing thread");
	destroyProcessingThread();

	puts("Bye bye!");
	exit(0);
}

int main(int argc, char const *argv[]){

	
	#ifdef ON_WINDOWS
		SetConsoleCtrlHandler(consoleHandler, true);
	#else
		struct sigaction sigIntHandler;

		sigIntHandler.sa_handler = sigintHandler;
		sigemptyset(&sigIntHandler.sa_mask);
		sigIntHandler.sa_flags = 0;

		sigaction(SIGINT, &sigIntHandler, NULL);
	#endif

	if (!NDIlib_initialize()) return 0;

	ndiFinder = NDIlib_find_create_v2();
	if (!ndiFinder) return 0;

	uint32_t ndiSourcesNo = 0;
	const NDIlib_source_t* ndiSources = NULL;
	while (!ndiSourcesNo) {
		print("Looking for NDI sources...\n");
		NDIlib_find_wait_for_sources(ndiFinder, 1000);
		ndiSources = NDIlib_find_get_current_sources(ndiFinder, &ndiSourcesNo);
		//TODO obtain source by name
		checkShutdown();
	}

	printf("Connecting to: %s\n", ndiSources[0].p_ndi_name);
	NDIlib_send_create_t ndiSendOpt;
	ndiSendOpt.p_ndi_name = "RealTime LLF out";
	ndiSendOpt.clock_audio = false;
	ndiSendOpt.clock_video = true;
	ndiSender = NDIlib_send_create();
	if (!ndiSender) return 0;
	NDIlib_recv_create_v3_t ndiRecvOpt;
	ndiRecvOpt.source_to_connect_to = ndiSources[0];
	ndiRecvOpt.p_ndi_recv_name = "RealTime LLF in";
	ndiRecvOpt.color_format = NDIlib_recv_color_format_RGBX_RGBA;
	ndiReceiver = NDIlib_recv_create_v3(&ndiRecvOpt);
	if (!ndiReceiver) return 0;

	NDIlib_find_destroy(ndiFinder); ndiFinder = NULL;
	#if CUDA_VERSION
		startGpuProcessingThread(0.35, 0.4, 5, 2, 256, 512);
	#elif OPENMP_VERSION
		startGpuProcessingThread(0.35, 0.4, 5, 3, 22);
	#else
		startGpuProcessingThread(0.35, 0.4, 5, 3);
	#endif

	NDIlib_video_frame_v2_t *ndiVideoFrame = new NDIlib_video_frame_v2_t;
	while(true){
		switch (NDIlib_recv_capture_v2(ndiReceiver, ndiVideoFrame, NULL, NULL, 5000)) {

			case NDIlib_frame_type_video: {
				//printff("Video data received (%dx%d).\n", ndiVideoFrame.xres, ndiVideoFrame.yres);
				handleIncomingFrame(ndiVideoFrame);
				writeOutputFrame(ndiVideoFrame);
				NDIlib_send_send_video_v2(ndiSender, ndiVideoFrame);
				NDIlib_recv_free_video_v2(ndiReceiver, ndiVideoFrame);
				break;
			}

			case NDIlib_frame_type_audio: break;
			case NDIlib_frame_type_none: {
				print("No data received.");
				break;
			}
			default: {
				print("Falling on default case");
				break;
			}
		}
		checkShutdown();
	}

	return 0;
}