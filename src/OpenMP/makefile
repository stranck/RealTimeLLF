GCC := g++
ARGS := -O3 -lm -g -ggdb -fopenmp
LINKER_ARGS := -lm
BINFOLDR := ../../bin/
UTILS := ../utils/

openmp : openmp.cpp openmpUtils.cpp $(UTILS)imageutils.cpp $(UTILS)utils.cpp $(UTILS)llfUtils.cpp
	$(GCC) $(ARGS) $^ $(LINKER_ARGS) -o $(BINFOLDR)$@

	