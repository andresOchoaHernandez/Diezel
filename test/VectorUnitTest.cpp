#include <iostream>
#include <cassert>
#include <limits>

#include "LinearAlgebra.hpp"
#include "testUtils.cpp"

int main()
{
    using LinearAlgebra::Vector;

    Vector a{100};
    a.randomInit(2,4);

    std::cout << a;

    return 0;
}