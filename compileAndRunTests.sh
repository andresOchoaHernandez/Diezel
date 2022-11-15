#!/bin/bash

nvcc  -arch=sm_75 --compiler-options "-fopenmp -Wall -Wextra -Werror" test/VectorUnitTest.cpp    src/Vector.cu src/Matrix.cu src/CSRMatrix.cu test/MeasureTime.cpp -I include/ -o VectorUnitTest
nvcc  -arch=sm_75 --compiler-options "-fopenmp -Wall -Wextra -Werror" test/MatrixUnitTest.cpp    src/Vector.cu src/Matrix.cu src/CSRMatrix.cu test/MeasureTime.cpp -I include/ -o MatrixUnitTest
nvcc  -arch=sm_75 --compiler-options "-fopenmp -Wall -Wextra -Werror" test/CSRMatrixUnitTest.cpp src/Vector.cu src/Matrix.cu src/CSRMatrix.cu test/MeasureTime.cpp -I include/ -o CSRMatrixUnitTest

./VectorUnitTest
rm VectorUnitTest
./MatrixUnitTest
rm MatrixUnitTest
./CSRMatrixUnitTest
rm CSRMatrixUnitTest