#include <cassert>

#include "LinearAlgebra.hpp"

int main()
{
    using LinearAlgebra::CSRMatrix;

    CSRMatrix a {10,10,10};
    a.randomInit(5,5);

    return 0;
}