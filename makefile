GCC := gcc
ARGS := -O3 -lm -g
SRC := src/
REAL_TIME_NDI := $(SRC)RealTime-NDI/
OPEN_MP := $(SRC)OpenMP/
CUDA := $(SRC)cuda/
LLF := $(SRC)llf/
UTILS := $(SRC)utils/
SCRIPTS := scripts/

.PHONY: default
default : all

.PHONY : clean
clean :
	rm -rf bin/
	rm -f tmp/*
	$(MAKE) -C $(LLF) clean
	$(MAKE) -C $(CUDA) clean
	$(MAKE) -C $(OPEN_MP) clean
	$(MAKE) -C $(REAL_TIME_NDI) clean

.PHONY : bin
bin :
	mkdir -p bin

.PHONY : openmp
openmp : testimage bin
	$(MAKE) -C $(OPEN_MP)

.PHONY : cuda
cuda : testimage bin
	$(MAKE) -C $(CUDA)

.PHONY : realtime-ndi
realtime-ndi : realtime-ndi-cuda realtime-ndi-openmp realtime-ndi-llf
.PHONY : realtime-ndi-cuda
realtime-ndi-cuda : bin
	$(MAKE) -C $(REAL_TIME_NDI) cuda
.PHONY : realtime-ndi-openmp
realtime-ndi-openmp : bin
	$(MAKE) -C $(REAL_TIME_NDI) openmp
.PHONY : realtime-ndi-llf
realtime-ndi-llf : bin
	$(MAKE) -C $(REAL_TIME_NDI) llf

.PHONY : llf
llf : testimage bin
	$(MAKE) -C $(LLF)

.PHONY : testimage
testimage :
	python3 $(SCRIPTS)staticImageConvertI2C.py imgTest/flower.png $(UTILS)test/testimage.h

all : clean bin openmp cuda llf realtime-ndi
	@echo "Compiled everything successfully"