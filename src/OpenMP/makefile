GCC := g++
COMPILER_ARGS := -O3 -lm -g -ggdb -fopenmp -DSHOW_TIME_STATS=1
COMPILER-OUTPUT_ARGS := -o
LINKER_ARGS := -lm
BINFOLDR := ../../bin/
UTILS := ../utils/
OUTPUT-EXTENSION :=

ifeq ($(OS),Windows_NT)
	GCC := cl
	COMPILER_ARGS := /Ox /openmp /DSHOW_TIME_STATS=1
	COMPILER-OUTPUT_ARGS:= /link /OUT:
	LINKER_ARGS :=
	OUTPUT-EXTENSION := .exe
endif

.PHONY: default
default: openmp clean;

.PHONY: clean
clean:
	rm -f utils.obj
	rm -f openmp.obj
	rm -f openmpMain.obj
	rm -f imageutils.obj
	rm -f openmpUtils.obj

openmp : openmpMain.cpp openmp.cpp openmpUtils.cpp $(UTILS)imageutils.cpp $(UTILS)utils.cpp $(UTILS)llfUtils.cpp
	$(GCC) $(COMPILER_ARGS) $^ $(LINKER_ARGS) $(COMPILER-OUTPUT_ARGS)$(BINFOLDR)$@$(OUTPUT-EXTENSION)

	