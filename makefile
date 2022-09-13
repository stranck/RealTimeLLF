GCC := gcc
ARGS := -O3 -lm -g
SRC := src/
REAL_TIME_NDI := $(SRC)RealTime-NDI/
OPEN_MP := $(SRC)OpenMP/
CUDA := $(SRC)cuda/
LLF := $(SRC)llf/
UTILS := $(SRC)utils/
SCRIPTS := scripts/

complete : clean all

.PHONY : clean
clean :
	rm -rf bin/
	rm -f tmp/*
	$(MAKE) -C $(CUDA) clean
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
realtime-ndi : bin
	$(MAKE) -C $(REAL_TIME_NDI)

.PHONY : llf
llf : testimage bin
	$(MAKE) -C $(LLF)

.PHONY : testimage
testimage :
	python3 $(SCRIPTS)staticImageConvertI2C.py imgTest/flower.png $(UTILS)test/testimage.h

all : clean bin openmp cuda llf realtime-ndi