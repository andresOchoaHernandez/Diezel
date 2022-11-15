#include <cassert>
#include <iostream>

#include "LinearAlgebra.hpp"
#include "MeasureTime.hpp"

void test_equlityOperator()
{
    //TODO:
}

void test_randomInit()
{
    //TODO:
}

void test_valInit()
{
    //TODO:
}

void test_vectorArithmethic()
{
    //TODO:
}

void test_constantArithmethic()
{
    //TODO:
}

void test_gpuArithmethic()
{
    //TODO:
}

int main()
{
    std::cout << "Vector test" << std::endl;

    using LinearAlgebra::Vector;

    Vector a{5};

    a.valInit(1);

    test_equlityOperator();
    test_randomInit();
    test_valInit();
    test_vectorArithmethic();
    test_constantArithmethic();
    test_gpuArithmethic();
    
    return 0;
}