GCC := gcc
ARGS := -O2 -lm -g
BINFOLDR := ../../bin/
UTILS := ../utils/

llf : llf.c $(UTILS)structs.h $(UTILS)imageutils.h $(UTILS)vects.h $(UTILS)utils.h $(UTILS)test/testimage.h
	$(GCC) $(ARGS) $^ -o $(BINFOLDR)$@