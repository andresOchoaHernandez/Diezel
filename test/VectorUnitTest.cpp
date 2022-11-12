#include <cassert>
#include <chrono>
#include <iostream>

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

void test_arithmetic()
{
    using LinearAlgebra::Vector;

    Vector a{10000000};
    a.valInit(1);
    Vector b{10000000};
    b.valInit(1);
    Vector c{10000000};
    c.valInit(2);

    Vector d = a + b;
    assert(c == d);

    Vector r1 = a + 1;
    assert(c == r1);

}

int main()
{
    //test_equalityOperator();
    test_arithmetic();

    return 0;
}