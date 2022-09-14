GCC := g++
GCC-ARGS := -O3 -g -lndi -pthread
GCC-OBJECT_ARGS := -c
GCC-OUTPUT_ARGS := -o
GCC-OBJECT-EXTENSION := .o
NVCC := nvcc
NVCC-ARGS := -O3 -arch=sm_60 -lineinfo -g -lndi #-dopt off
NVCC-OBJECT_ARGS := -dc
NVCC-OUTPUT_ARGS := -o

BINFOLDR := ../../bin/
UTILS := ../utils/
CUDA := ../CUDA/

ifeq ($(OS),Windows_NT)
	GCC := cl
	GCC-ARGS := /Ox #/Zi #idk why this is not working :/ 
	GCC-OBJECT_ARGS := /c
	GCC-OUTPUT_ARGS:= /OUT:
	GCC-OBJECT-EXTENSION := .obj
endif

.PHONY: default
default: all;

cudaUtils.o: $(CUDA)cudaUtils.cu 
	$(NVCC) $(NVCC-ARGS) $(NVCC-OBJECT_ARGS) $^ $(NVCC-OUTPUT_ARGS) $@
cuda.o: $(CUDA)cuda.cu
	$(NVCC) $(NVCC-ARGS) $(NVCC-OBJECT_ARGS) $^ $(NVCC-OUTPUT_ARGS) $@

realtime-ndi$(GCC-OBJECT-EXTENSION): realtime-ndi.cpp
	$(GCC) $(GCC-ARGS) $(GCC-OBJECT_ARGS) $^ $(GCC-OUTPUT_ARGS)$@
processingHandler$(GCC-OBJECT-EXTENSION): processingHandler.cpp
	$(GCC) $(GCC-ARGS) $(GCC-OBJECT_ARGS) $^ $(GCC-OUTPUT_ARGS)$@

.PHONY: clean
clean:
	rm -f processingHandler.exp
	rm -f processingHandler.lib
	rm -f processingHandler.pdb
	rm -f processingHandler.obj
	rm -f processingHandler.o
	rm -f realtime-ndi.pdb
	rm -f realtime-ndi.obj
	rm -f realtime-ndi.o
	rm -f cudaUtils.o
	rm -f vc140.pdb
	rm -f cuda.o

.PHONY: realtime-ndi
realtime-ndi-cuda: cudaUtils.o cuda.o processingHandler$(GCC-OBJECT-EXTENSION) realtime-ndi$(GCC-OBJECT-EXTENSION) $(UTILS)imageutils.cpp $(UTILS)utils.cpp
	$(NVCC) $(NVCC-ARGS) $^ $(NVCC-OUTPUT_ARGS) $(BINFOLDR)$@

.PHONY: all
all: realtime-ndi-cuda clean