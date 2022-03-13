GCC := gcc
COMPILER_ARGS := -O2 -g
LINKER_ARGS := -lm
BINFOLDR := ../../bin/
UTILS := ../utils/

llf : llf.c $(UTILS)imageutils.c $(UTILS)utils.c $(UTILS)vects.c $(UTILS)llfUtils.c
	$(GCC) $(COMPILER_ARGS) $^ $(LINKER_ARGS) -o $(BINFOLDR)$@