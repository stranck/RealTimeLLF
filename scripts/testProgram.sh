#!/bin/sh

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 [llf|cuda|openmp]" >&2
  exit 1
fi

make $1
mkdir tmp
rm tmp/testStaticIO.out.txt
rm tmp/testStaticIO.out.png
if [ "$1" = "cuda" ]; then
  bin/$1 512 256 > tmp/testStaticIO.out.txt
elif [ "$1" = "openmp" ]; then
  bin/$1 24 > tmp/testStaticIO.out.txt
else
  bin/$1 > tmp/testStaticIO.out.txt
fi
python3 scripts/staticImageConvertT2I.py tmp/testStaticIO.out.txt tmp/testStaticIO.out.png
