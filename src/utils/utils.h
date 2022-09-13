#pragma once

#include <stdint.h>
#include <stdio.h>
#include <math.h>

#define dumpMemory(data, len){printBuffer((uint8_t *) data, len); puts(""); fflush(stdout);}

void printBuffer(uint8_t *data, uint32_t len);

int32_t roundfI32(float f);
uint8_t roundfu8(float f);

#define print(str){puts(str); fflush(stdout);}
#define printff(format, ...){fprintf(stderr, format, __VA_ARGS__); fflush(stderr);}

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
	#define ON_WINDOWS 1
#endif

#ifdef _MSC_BUILD
	#include <malloc.h>
	#define allocStack(size)(_alloca(size))
#else
	#define allocStack(size)(alloca(size))
#endif

typedef uint64_t TimeCounter;
#ifdef SHOW_TIME_STATS
	#ifdef ON_WINDOWS
		#include <sys/timeb.h> 

		typedef timeb TimeData;
		#define startTimerCounter(timeData){ftime(&(timeData));}
		#define stopTimerCounter(timeData, timeCounter){ \
			TimeCounter _startTime = (timeData).time; \
			TimeCounter _startMilli = (timeData).millitm; \
			startTimerCounter(timeData); \
			(timeCounter) += (TimeCounter) (1000.0 * ((timeData).time - _startTime) + ((timeData).millitm - _startMilli)); \
		}
	#else
		#include <sys/time.h>

		typedef timeval TimeData;
		#define startTimerCounter(timeData){gettimeofday(&(timeData), NULL);}
		#define stopTimerCounter(timeData, timeCounter){ \
			TimeCounter _startTvsec = (timeData).tv_sec; \
			TimeCounter _startTvusec = (timeData).tv_usec; \
			startTimerCounter(timeData); \
			(timeCounter) += (((timeData).tv_sec - _startTvsec) * 1000000 + (timeData).tv_usec - _startTvusec) / 1000; \
		}
	#endif
#else
	#define TimeData uint8_t
	#define startTimerCounter(timeData){}
	#define stopTimerCounter(timeData, timeCounter){}
#endif