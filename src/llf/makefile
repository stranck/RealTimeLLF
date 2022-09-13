GCC := gcc
COMPILER_ARGS := -O3 -g -DSHOW_TIME_STATS=1
LINKER_ARGS := -lm
BINFOLDR := ../../bin/
UTILS := ../utils/

llf : llf.cpp $(UTILS)imageutils.cpp $(UTILS)utils.cpp $(UTILS)llfUtils.cpp
	$(GCC) $(COMPILER_ARGS) $^ $(LINKER_ARGS) -o $(BINFOLDR)$@