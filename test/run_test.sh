#!/bin/bash
nvcc -w -std=c++11 -arch=sm_75 --compiler-options "-Wall -Wextra -Wpedantic -Werror -O3" test.cu ../gpuLinAlg.cu -I../ -o test
./test
rm test