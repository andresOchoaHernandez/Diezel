#include <iostream>
#include <cassert>
#include <limits>

#include "gpuLinAlg.hpp"
#include "testUtils.cpp"

int main()
{
    using gpuLinAlg::Vector;

    Vector a{100};
    a.randomInit(2,4);

    printVector(a);

    return 0;
}