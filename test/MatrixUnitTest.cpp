#include <iostream>
#include <cassert>
#include <limits>

#include "LinearAlgebra.hpp"
#include "testUtils.cpp"

int main()
{
    using LinearAlgebra::Matrix;

    Matrix a{10,10};
    a.randomInit(1,10);

    std::cout << a;
    
    return 0;
}