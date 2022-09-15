GCC := gcc
COMPILER_ARGS := -O3 -g -DSHOW_TIME_STATS=1
COMPILER-OUTPUT_ARGS := -o
LINKER_ARGS := -lm
BINFOLDR := ../../bin/
UTILS := ../utils/
OUTPUT-EXTENSION :=

ifeq ($(OS),Windows_NT)
	GCC := cl
	COMPILER_ARGS := /Ox /DSHOW_TIME_STATS=1
	COMPILER-OUTPUT_ARGS:= /link /OUT:
	LINKER_ARGS :=
	OUTPUT-EXTENSION := .exe
endif

.PHONY: default
default: llf clean;

.PHONY: clean
clean:
	rm -f llf.obj
	rm -f utils.obj
	rm -f mainLlf.obj
	rm -f llfUtils.obj
	rm -f imageutils.obj

llf : mainLlf.cpp llf.cpp $(UTILS)imageutils.cpp $(UTILS)utils.cpp $(UTILS)llfUtils.cpp
	$(GCC) $(COMPILER_ARGS) $^ $(LINKER_ARGS) $(COMPILER-OUTPUT_ARGS)$(BINFOLDR)$@$(OUTPUT-EXTENSION)