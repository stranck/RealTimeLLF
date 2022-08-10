GCC := nvcc
ARGS := -O0 -lm #-dopt off
BINFOLDR := ../../bin/
UTILS := ../utils/

cuda : cuda.cu cudaUtils.cu $(UTILS)imageutils.cpp $(UTILS)utils.cpp $(UTILS)vects.cpp $(UTILS)llfUtils.cpp
	$(GCC) $(ARGS) $^ -o $(BINFOLDR)$@