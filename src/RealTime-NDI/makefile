GCC := g++
NVCC := nvcc
GCC-ARGS := -O0 -g -lndi -std=c++2a -pthread
NVCC-ARGS := -O0 -arch=sm_70 -lineinfo -g -lndi #-dopt off
BINFOLDR := ../../bin/
UTILS := ../utils/
CUDA := ../CUDA/

.PHONY: default
default: all;

cudaUtils.o: $(CUDA)cudaUtils.cu 
	$(NVCC) $(NVCC-ARGS) -dc $^ -o $@
cuda.o: $(CUDA)cuda.cu
	$(NVCC) $(NVCC-ARGS) -dc $^ -o $@

realtime-ndi.o: realtime-ndi.cpp
	$(NVCC) $(NVCC-ARGS) -dc $^ -o $@
cudaHandler.o: cudaHandler.cu
	$(NVCC) $(NVCC-ARGS) -dc $^ -o $@

.PHONY: clean
clean:
	rm -f cudaHandler.exp
	rm -f cudaHandler.lib
	rm -f cudaHandler.pdb
	rm -f realtime-ndi.pdb
	rm -f realtime-ndi.o
	rm -f cudaHandler.o
	rm -f cudaUtils.o
	rm -f vc140.pdb
	rm -f cuda.o

.PHONY: realtime-ndi
realtime-ndi-cuda: cudaUtils.o cuda.o cudaHandler.o realtime-ndi.o $(UTILS)imageutils.cpp $(UTILS)utils.cpp
	$(NVCC) $(NVCC-ARGS) $^ -o $(BINFOLDR)$@

.PHONY: all
all: realtime-ndi-cuda #clean