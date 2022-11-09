#include <cassert>

#include "LinearAlgebra.hpp"

void test_equalityOperator()
{
    using LinearAlgebra::Matrix;

    Matrix a{10,10};
    a.valInit(1);

    Matrix b{10,10};
    b.valInit(1);

    assert(a == b);

    b.valInit(2);

    assert(!(a == b));

    Matrix c{2,2};
    c.valInit(1);

    assert(!(a == c));
}

int main()
{
    test_equalityOperator();
    
    return 0;
}