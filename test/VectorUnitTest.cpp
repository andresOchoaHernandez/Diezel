#include <cassert>

#include "LinearAlgebra.hpp"

void test_equalityOperator()
{
    using LinearAlgebra::Vector;

    Vector a{100};
    a.valInit(1);

    Vector b{100};
    b.valInit(1);

    assert(a == b);

    b.valInit(2);

    assert(!(a == b));

    Vector c{10};
    c.valInit(1);

    assert(!(a == c));
}

int main()
{
    test_equalityOperator();

    return 0;
}