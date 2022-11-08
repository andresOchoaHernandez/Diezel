#include <iostream>
#include <cassert>
#include <limits>

#include "gpuLinAlg.hpp"
#include "testUtils.cpp"

int main()
{
    using gpuLinAlg::Matrix;

    Matrix a{10,10};
    a.randomInit(1,10);

    printMatrix(a);

    return 0;
}