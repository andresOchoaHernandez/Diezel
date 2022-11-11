#include <cassert>
#include <iostream>

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

void test_arithmetic()
{
    using LinearAlgebra::Matrix;

    const unsigned rows = 1200;
    const unsigned cols = 1200;

    Matrix a{rows,cols};
    a.valInit(1);

    Matrix b{rows,cols};
    b.valInit(1);

    Matrix c = a + b;

    Matrix d{rows,cols};
    d.valInit(2);

    assert(c == d);

    Matrix f{rows,cols};
    f.valInit(2);

    Matrix g = a + 1;
    assert(f == g);
}

int main()
{
    test_equalityOperator();
    test_arithmetic();
    
    return 0;
}