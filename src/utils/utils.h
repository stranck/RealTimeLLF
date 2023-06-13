#pragma once

#include <stdint.h>
#include <stdio.h>
#include <math.h>

/**
 * @brief wrapper of printBuffer, to easily dump buffers
 */
#define dumpMemory(data, len){printBuffer((uint8_t *) data, len); puts(""); fflush(stdout);}

void printBuffer(uint8_t *data, uint32_t len);

/**
 * @brief round to nearest integer, away from zero, and cast it into a int32_t
 * 
 * @param f float to round
 * @return int32_t f rounded and casted
 */
inline int32_t roundfI32(float f){
	return (int32_t) roundf(f);
}
/**
 * @brief round to nearest integer, away from zero, and cast it into a uint8_t
 * 
 * @param f float to round
 * @return uint8_t f rounded and casted
 */
inline uint8_t roundfu8(float f){
	return (uint8_t) roundf(f);
}

/**
 * @brief Wrapper of puts() that also flushes stdout
 */
#define print(str){puts(str); fflush(stdout);}
#define LOG_FILE_DESCRIPTOR stderr
/**
 * @brief Wrapper of printf() that uses and also flushes LOG_FILE_DESCRIPTOR
 */
#define printff(format, ...){fprintf(LOG_FILE_DESCRIPTOR, format, __VA_ARGS__); fflush(LOG_FILE_DESCRIPTOR);}

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
	#define ON_WINDOWS 1
#endif

//Wrapper of alloca() that automatically uses the correct version based on the platform
#ifdef _MSC_BUILD
	#include <malloc.h>
	/**
	 * @brief platform independent alloca() wrapper
	 */
	#define allocStack(size)(_alloca(size))
#else
	/**
	 * @brief platform independent alloca() wrapper
	 */
	#define allocStack(size)(alloca(size))
#endif

typedef uint64_t TimeCounter;
#if defined(SHOW_TIME_STATS) || defined(SHOW_TIME_STATS_NDI)
	//Switch time implementation based on the current platform
	#ifdef ON_WINDOWS
		#include <sys/timeb.h> 

		typedef timeb TimeData;
		/**
		 * @brief start counting time
		 */
		#define startTimerCounter(timeData){ftime(&(timeData));}
		/**
		 * @brief stops counting time and adds the elapsed time to the counter
		 */
		#define stopTimerCounter(timeData, timeCounter){ \
			TimeCounter _startTime = (timeData).time; \
			TimeCounter _startMilli = (timeData).millitm; \
			startTimerCounter(timeData); \
			(timeCounter) += (TimeCounter) (1000.0 * ((timeData).time - _startTime) + ((timeData).millitm - _startMilli)); \
		}
	#else
		#include <sys/time.h>

		typedef timeval TimeData;
		/**
		 * @brief start counting time
		 */
		#define startTimerCounter(timeData){gettimeofday(&(timeData), NULL);}
		/**
		 * @brief stops counting time and adds the elapsed time to the counter
		 */
		#define stopTimerCounter(timeData, timeCounter){ \
			TimeCounter _startTvsec = (timeData).tv_sec; \
			TimeCounter _startTvusec = (timeData).tv_usec; \
			startTimerCounter(timeData); \
			(timeCounter) += (((timeData).tv_sec - _startTvsec) * 1000000 + (timeData).tv_usec - _startTvusec) / 1000; \
		}
	#endif
#else
	//If we don't want to display time stats on each render, we define empty macros
	#define TimeData uint8_t
	/**
	 * @brief start counting time
	 */
	#define startTimerCounter(timeData){}
	/**
	 * @brief stops counting time and adds the elapsed time to the counter
	 */
	#define stopTimerCounter(timeData, timeCounter){}
#endif