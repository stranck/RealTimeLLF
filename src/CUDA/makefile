GCC := nvcc
ARGS := -O0 -lm -arch=sm_70 -g #-dopt off
BINFOLDR := ../../bin/
UTILS := ../utils/

.PHONY: default
default: all;

cudaUtils.o: cudaUtils.cu 
	$(GCC) $(ARGS) -dc $^ -o $@
bufferManager.o: bufferManager.cu
	$(GCC) $(ARGS) -dc $^ -o $@
cuda.o: cuda.cu
	$(GCC) $(ARGS) -dc $^ -o $@

.PHONY: clean
clean:
	rm -f cudaUtils.o
	rm -f bufferManager.o
	rm -f cuda.o

.PHONY: cuda
cuda: cuda.o cudaUtils.o bufferManager.o $(UTILS)imageutils.cpp $(UTILS)utils.cpp
	$(GCC) $(ARGS) $^ -o $(BINFOLDR)$@

.PHONY: all
all: cuda #clean