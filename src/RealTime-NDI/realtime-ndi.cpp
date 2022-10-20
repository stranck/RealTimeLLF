#include "realtime-ndi.h"

//Notify all threads to stop their activity when a Ctrl+c or sigint is detected by the process
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

/**
 * @brief Shutdowns the process by destroying ndi's stuff and notify the processing thread to end its activity
 */
void cleanup(){
	print("Shutting down NDI");
	if(ndiReceiver) NDIlib_recv_destroy(ndiReceiver);
	if(ndiFinder) NDIlib_find_destroy(ndiFinder);
	if(ndiSender) NDIlib_send_destroy(ndiSender);
	NDIlib_destroy();
	print("Shuttind down processing thread");
	destroyProcessingThread(); //synchronous call

	print("Bye bye!");
	exit(0);
}

//0.35 0.4 5 3 512 256
/**
 * @brief Argouments to the main realtime-ndi method: <ndi-source name> <sigma> <alpha> <beta> <nLevels> <number of blocks> <number of threads>
 */
int main(int argc, char const *argv[]){
	//loads version specific args
	#if CUDA_VERSION
		if(argc < 8){
			printff("Usage: %s <ndi-source name> <sigma> <alpha> <beta> <nLevels> <number of blocks> <number of threads>\n", argv[0]);
			exit(1);
		}
		uint32_t nBlocks = atoi(argv[6]);
		printff("nBlocks: %d\n", nBlocks);
		uint32_t nThreads = atoi(argv[7]);
		printff("nThreads: %d\n", nThreads);
	#elif OPENMP_VERSION
		if(argc < 7){
			printff("Usage: %s <ndi-source name> <sigma> <alpha> <beta> <nLevels> <number of threads>\n", argv[0]);
			exit(1);
		}
		uint8_t nThreads = atoi(argv[6]);
	#else
		if(argc < 6){
			printff("Usage: %s <ndi-source name> <sigma> <alpha> <beta> <nLevels>\n", argv[0]);
			exit(1);
		}
	#endif
	//load general args
	const char *ndiSourceName = argv[1];
	printff("ndiSourceName: %s\n", ndiSourceName);
	float sigma = atof(argv[2]), alpha = atof(argv[3]), beta = atof(argv[4]);
	printff("sigma: %f    alpha: %f    beta%f\n", sigma, alpha, beta);
	uint8_t nLevels = atoi(argv[5]);
	printff("nLevels: %d\n", nLevels);
	
	//adds ctrl+c/sigint handler
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

	ndiFinder = NDIlib_find_create_v2(); //Creates an ndi source finder
	if (!ndiFinder) return 0;

	uint32_t ndiSourcesNo = 0;
	const NDIlib_source_t *choosed = NULL;
	const NDIlib_source_t* ndiSources = NULL;
	while (choosed == NULL) { //Search for the ndi source by name
		print("Looking for NDI sources...\n");
		NDIlib_find_wait_for_sources(ndiFinder, 2500);
		ndiSources = NDIlib_find_get_current_sources(ndiFinder, &ndiSourcesNo); //Get the current ndi sources
		checkShutdown(); //Checks if a process shutdown has been requested. If yes, the program is going to call exit(0)
		for(uint32_t i = 0; i < ndiSourcesNo; i++){ //For each ndi source
			printff("Detected NDI source: '%s'\n", ndiSources[i].p_ndi_name);
			if(!strcmp(ndiSources[i].p_ndi_name, ndiSourceName)){ //Check if its name is the chosen one
				choosed = &ndiSources[i];
				break; //If yes, breaks and continue with the main method
			}
		}
		print("");
	}

	printff("Connecting to: %s\n", ndiSources[0].p_ndi_name);
	//Creates an ndi send object specifying its name
	NDIlib_send_create_t ndiSendOpt;
	ndiSendOpt.p_ndi_name = "RealTime LLF out";
	ndiSendOpt.clock_audio = false;
	ndiSendOpt.clock_video = true;
	ndiSender = NDIlib_send_create(&ndiSendOpt);
	if (!ndiSender) return 0;
	//Creates an ndi receive object specifying the source to connect to, the receive name and sets the encoding of the incoming video frames to RGBX/A (otherwise it's going to use UYVY)
	NDIlib_recv_create_v3_t ndiRecvOpt;
	ndiRecvOpt.source_to_connect_to = *choosed;
	ndiRecvOpt.p_ndi_recv_name = "RealTime LLF in";
	ndiRecvOpt.color_format = NDIlib_recv_color_format_RGBX_RGBA;
	ndiReceiver = NDIlib_recv_create_v3(&ndiRecvOpt);
	if (!ndiReceiver) return 0;

	NDIlib_find_destroy(ndiFinder); ndiFinder = NULL; //destroys the ndi finder object
	//Starts the processing thread
	#if CUDA_VERSION
		startProcessingThread(sigma, alpha, beta, nLevels, nBlocks, nThreads);
	#elif OPENMP_VERSION
		startProcessingThread(sigma, alpha, beta, nLevels, nThreads);
	#else
		startProcessingThread(sigma, alpha, beta, nLevels);
	#endif

	//allocates a buffer ndiVideoFrame
	NDIlib_video_frame_v2_t *ndiVideoFrame = new NDIlib_video_frame_v2_t;
	while(true){
		switch (NDIlib_recv_capture_v2(ndiReceiver, ndiVideoFrame, NULL, NULL, 5000)) { //Capture an ndi packet

			case NDIlib_frame_type_video: { //if the ndi packet is a video packet
				//printff("Video data received (%dx%d).\n", ndiVideoFrame.xres, ndiVideoFrame.yres);
				handleIncomingFrame(ndiVideoFrame); //pass it to the processing thread
				getOutputFrame(ndiVideoFrame); //get the last rendered output from the processing thread
				NDIlib_send_send_video_v2(ndiSender, ndiVideoFrame); //send the rendered frame
				NDIlib_recv_free_video_v2(ndiReceiver, ndiVideoFrame); //clear the ndi buffers
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
		checkShutdown(); //Checks if a process shutdown has been requested. If yes, the program is going to call exit(0)
	}

	return 0;
}