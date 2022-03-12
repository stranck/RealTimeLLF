GCC := gcc
ARGS := -O2 -lm -g
BINFOLDR := ../../bin/
UTILS := ../utils/

cuda : cuda.c $(UTILS)test.c
	$(GCC) $(ARGS) $^ -o $(BINFOLDR)$@