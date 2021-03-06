#!/bin/sh

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 [llf|cuda|openmp]" >&2
  exit 1
fi

make $1
mkdir tmp
bin/$1 > tmp/testStaticIO.out.txt
python3 scripts/staticImageConvertT2I.py tmp/testStaticIO.out.txt tmp/testStaticIO.out.png