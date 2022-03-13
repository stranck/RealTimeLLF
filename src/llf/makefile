GCC := gcc
ARGS := -O2 -lm -g
BINFOLDR := ../../bin/
UTILS := ../utils/

llf : llf.c $(UTILS)imageutils.c $(UTILS)utils.c $(UTILS)vects.c
	$(GCC) $(ARGS) $^ -o $(BINFOLDR)$@