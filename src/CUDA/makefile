GCC := nvcc
ARGS := -O3 -arch=sm_60 -lineinfo -g -DSHOW_TIME_STATS=1 #-dopt off
BINFOLDR := ../../bin/
UTILS := ../utils/

.PHONY: default
default: all;

cudaUtils.o: cudaUtils.cu 
	$(GCC) $(ARGS) -dc $^ -o $@
cuda.o: cuda.cu
	$(GCC) $(ARGS) -dc $^ -o $@
mainCuda.o: mainCuda.cu
	$(GCC) $(ARGS) -dc $^ -o $@

.PHONY: clean
clean:
	rm -f vc140.pdb
	rm -f bufferManager.o
	rm -f cudaUtils.o
	rm -f mainCuda.o
	rm -f cuda.o

.PHONY: cuda
cuda: mainCuda.o cuda.o cudaUtils.o $(UTILS)imageutils.cpp $(UTILS)utils.cpp
	$(GCC) $(ARGS) $^ -o $(BINFOLDR)$@

.PHONY: all
all: cuda clean