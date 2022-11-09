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

void test_multi()
{
    using LinearAlgebra::Vector;

    Vector a{100000000};
    a.valInit(1);
    Vector b{100000000};
    b.valInit(1);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Vector c1 = a.seqVectorSum(b); 
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Sequential vector sum: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;


    begin = std::chrono::steady_clock::now();
    Vector c2 = a.threadedVectorSum(b);
    end = std::chrono::steady_clock::now();
    std::cout << "threaded vector sum: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
    assert(c1 == c2);
}

int main()
{
    test_equalityOperator();
    test_multi();

    return 0;
}