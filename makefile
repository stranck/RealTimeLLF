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

.PHONY : bin
bin :
	mkdir -p bin

.PHONY : openmp
openmp :
	$(MAKE) -C $(OPEN_MP)

.PHONY : cuda
cuda :
	$(MAKE) -C $(CUDA)

.PHONY : llf
llf : testimage
	$(MAKE) -C $(LLF)

.PHONY : testimage
testimage :
	python3 $(SCRIPTS)staticImageConvert.py imgTest/razzi.jpg $(UTILS)test/testimage.h

all : bin openmp cuda llf