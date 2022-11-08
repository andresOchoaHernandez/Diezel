#include <iostream>
#include <cassert>
#include <limits>

#include "gpuLinAlg.hpp"
#include "testUtils.cpp"

int main()
{
    using gpuLinAlg::CSRMatrix;

    CSRMatrix a {5,10,10};
    a.randomInit(4,9);

    std::cout << a;

    return 0;
}