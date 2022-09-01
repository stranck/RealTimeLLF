GCC := gcc
ARGS := -O2 -lm -g
SRC := src/
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

.PHONY : bin
bin :
	mkdir -p bin

.PHONY : openmp
openmp : testimage bin
	$(MAKE) -C $(OPEN_MP)

.PHONY : cuda
cuda : testimage bin
	$(MAKE) -C $(CUDA)

.PHONY : llf
llf : testimage bin
	$(MAKE) -C $(LLF)

.PHONY : testimage
testimage :
	python3 $(SCRIPTS)staticImageConvertI2C.py imgTest/flower.png $(UTILS)test/testimage.h

all : bin openmp cuda llf