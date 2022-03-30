GCC := gcc
ARGS := -O3 -lm -g -ggdb -fopenmp
LINKER_ARGS := -lm
BINFOLDR := ../../bin/
UTILS := ../utils/

openmp : openmp.c openmpUtils.c $(UTILS)imageutils.c $(UTILS)utils.c $(UTILS)vects.c $(UTILS)llfUtils.c
	$(GCC) $(ARGS) $^ $(LINKER_ARGS) -o $(BINFOLDR)$@

	