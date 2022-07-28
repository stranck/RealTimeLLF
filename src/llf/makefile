GCC := gcc
COMPILER_ARGS := -O3 -g
LINKER_ARGS := -lm
BINFOLDR := ../../bin/
UTILS := ../utils/

llf : llf.cpp $(UTILS)imageutils.cpp $(UTILS)utils.cpp $(UTILS)vects.cpp $(UTILS)llfUtils.cpp
	$(GCC) $(COMPILER_ARGS) $^ $(LINKER_ARGS) -o $(BINFOLDR)$@