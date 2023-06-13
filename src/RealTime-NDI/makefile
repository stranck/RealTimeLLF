GCC := g++
GCC-ARGS := -O3 -g -lndi -pthread -fopenmp
GCC-OBJECT_ARGS := -c
GCC-OUTPUT_ARGS := -o
GCC-LINKER_ARGS := -lndi
GCC-PREPROC_ARGS := -D
GCC-OBJECT-EXTENSION := .o
NVCC := nvcc
NVCC-ARGS := -O3 -arch=sm_60 -lineinfo -g -lndi #-dopt off
NVCC-OBJECT_ARGS := -dc
NVCC-OUTPUT_ARGS := -o
NVCC-PREPROC_ARGS := -D
OUTPUT-EXTENSION :=

VERSION-NAME :=LLF_VERSION

BINFOLDR := ../../bin/
UTILS := ../utils/
CUDA := ../CUDA/
OPENMP := ../OpenMP/
LLF := ../llf/

ifeq ($(OS),Windows_NT)
	GCC := cl
	GCC-ARGS := /Ox /openmp #/Zi #idk why this is not working :/ 
	GCC-OBJECT_ARGS := /c
	GCC-OUTPUT_ARGS:= /link /OUT:
	GCC-LINKER_ARGS :=
	GCC-OBJECT-EXTENSION := .obj
	OUTPUT-EXTENSION := .exe
endif

.PHONY: default
default: help;

choose-cuda:
	$(eval VERSION-NAME :=CUDA_VERSION)
	@echo Choosed cuda
choose-openmp:
	$(eval VERSION-NAME :=OPENMP_VERSION)
	@echo Choosed openmp
choose-llf:
	$(eval VERSION-NAME :=LLF_VERSION)
	@echo Choosed llf

cudaUtils.o: $(CUDA)cudaUtils.cu 
	$(NVCC) $(NVCC-ARGS) $(NVCC-OBJECT_ARGS) $^ $(NVCC-OUTPUT_ARGS) $@
cuda.o: $(CUDA)cuda.cu
	$(NVCC) $(NVCC-ARGS) $(NVCC-OBJECT_ARGS) $^ $(NVCC-OUTPUT_ARGS) $@

realtime-ndi$(GCC-OBJECT-EXTENSION): realtime-ndi.cpp
	$(GCC) $(GCC-ARGS) $(GCC-OBJECT_ARGS) $(GCC-PREPROC_ARGS)$(VERSION-NAME)=1 $^ $(GCC-OUTPUT_ARGS)$@
processingHandler$(GCC-OBJECT-EXTENSION): processingHandler.cpp
	$(GCC) $(GCC-ARGS) $(GCC-OBJECT_ARGS) $(GCC-PREPROC_ARGS)$(VERSION-NAME)=1 $(GCC-PREPROC_ARGS)SHOW_TIME_STATS_NDI=1 $(GCC-PREPROC_ARGS)PRINT_STAT_EVERY_N_FRAMES=100 $^ $(GCC-OUTPUT_ARGS)$@

.PHONY: clean
clean:
	rm -f processingHandler$(GCC-OBJECT-EXTENSION)
	rm -f realtime-ndi$(GCC-OBJECT-EXTENSION)
	rm -f openmpUtils$(GCC-OBJECT-EXTENSION)
	rm -f imageutils$(GCC-OBJECT-EXTENSION)
	rm -f llfUtils$(GCC-OBJECT-EXTENSION)
	rm -f openmp$(GCC-OBJECT-EXTENSION)
	rm -f utils$(GCC-OBJECT-EXTENSION)
	rm -f llf$(GCC-OBJECT-EXTENSION)

	rm -f processingHandler.exp
	rm -f processingHandler.lib
	rm -f processingHandler.pdb
	rm -f realtime-ndi.pdb
	rm -f cudaUtils.o
	rm -f vc140.pdb
	rm -f cuda.o

realtime-ndi-cuda: cudaUtils.o cuda.o processingHandler$(GCC-OBJECT-EXTENSION) realtime-ndi$(GCC-OBJECT-EXTENSION) $(UTILS)imageutils.cpp $(UTILS)utils.cpp
	$(NVCC) $(NVCC-ARGS) $(NVCC-PREPROC_ARGS)CUDA_VERSION=1 $^ $(NVCC-OUTPUT_ARGS) $(BINFOLDR)$@

realtime-ndi-openmp: $(OPENMP)openmp.cpp $(OPENMP)openmpUtils.cpp processingHandler$(GCC-OBJECT-EXTENSION) realtime-ndi$(GCC-OBJECT-EXTENSION) $(UTILS)imageutils.cpp $(UTILS)utils.cpp $(UTILS)llfUtils.cpp
	$(GCC) $(GCC-ARGS) $(GCC-PREPROC_ARGS)OPENMP_VERSION=1 $^ $(GCC-OUTPUT_ARGS)$(BINFOLDR)$@$(OUTPUT-EXTENSION) $(GCC-LINKER_ARGS)

realtime-ndi-llf: $(LLF)llf.cpp processingHandler$(GCC-OBJECT-EXTENSION) realtime-ndi$(GCC-OBJECT-EXTENSION) $(UTILS)imageutils.cpp $(UTILS)utils.cpp $(UTILS)llfUtils.cpp 
	$(GCC) $(GCC-ARGS) $(GCC-PREPROC_ARGS)LLF_VERSION=1 $^ $(GCC-OUTPUT_ARGS)$(BINFOLDR)$@$(OUTPUT-EXTENSION) $(GCC-LINKER_ARGS)

.PHONY: cuda
cuda: choose-cuda realtime-ndi-cuda clean
.PHONY: openmp
openmp: choose-openmp realtime-ndi-openmp clean
.PHONY: llf
llf: choose-llf realtime-ndi-llf clean

.PHONY: help
help:
	@echo Compile with the options [cuda\|openmp\|llf]