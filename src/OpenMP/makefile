GCC := gcc
ARGS := -O3 -lm -g -fopenmp
BINFOLDR := ../../bin/
UTILS := ../utils/

openmp : openmp.c $(UTILS)test.c
	$(GCC) $(ARGS) $^ -o $(BINFOLDR)$@