GCC := nvcc
ARGS := -O0 -lm -arch=sm_70 #-dopt off
BINFOLDR := ../../bin/
UTILS := ../utils/

cuda : cuda.cu cudaUtils.cu bufferManager.cu $(UTILS)imageutils.cpp $(UTILS)utils.cpp $(UTILS)llfUtils.cpp
	$(GCC) $(ARGS) $^ -o $(BINFOLDR)$@