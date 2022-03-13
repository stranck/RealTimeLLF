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
openmp : bin
	$(MAKE) -C $(OPEN_MP)

.PHONY : cuda
cuda : bin
	$(MAKE) -C $(CUDA)

.PHONY : llf
llf : testimage bin
	$(MAKE) -C $(LLF)

.PHONY : testimage
testimage :
	python3 $(SCRIPTS)staticImageConvertI2C.py imgTest/razzi.jpg $(UTILS)test/testimage.h

all : bin openmp cuda llf