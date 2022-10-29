#include <iostream>
#include <cassert>
#include <limits>

#include "gpuLinAlg.hpp"

//TODO: write unit tests for all classes and methods

#define TEST_SIZE 800

#define ROWS    94340
#define COLUMNS 1024

void printMatrix(gpuLinAlg::Matrix& matrix)
{
    int* data = matrix.data();

    std::cout << "---------------------" << std::endl;

    for(unsigned i = 0u ; i < matrix.rows() ; i++)
    {
        for(unsigned j = 0u ; j < matrix.cols() ; j++)
        {
            std::cout << data[i*matrix.cols() + j] << " "; 
        }

        std::cout << std::endl;
    }

    std::cout << "---------------------" << std::endl;
}

void printVector(gpuLinAlg::Vector& v)
{
    std::cout << "---------------------" << std::endl;

    int* vec = v.getVec();

    for(unsigned i = 0u ; i < v.len() ; i++)
    {
        std::cout << vec[i] << " ";
    } 
    std::cout << std::endl;

    std::cout << "---------------------" << std::endl;
}

bool checkIfVectorAreEqual(gpuLinAlg::Vector& v1,gpuLinAlg::Vector& v2)
{
    int* v1Vec = v1.getVec();
    int* v2Vec = v2.getVec();

    for(unsigned i = 0u ; i < v1.len() ; i++ )
    {
        if(v1Vec[i] != v2Vec[i]) 
        {
            std::cout << "Error at index: " << i << std::endl
                      << "v1 : " << v1[i] << " " << "v2 : " << v2[i] << std::endl;

            return false;
        }
    }

    return true;
}

void test1()
{
    gpuLinAlg::Vector t1_v1{TEST_SIZE};
    gpuLinAlg::Vector t1_v2{TEST_SIZE};

    t1_v1.randomInit(0,9);
    t1_v2.randomInit(0,5);

    gpuLinAlg::Vector t1_gpuRv = t1_v1.gpuVectorDif(t1_v2);
    gpuLinAlg::Vector t1_seqRv = t1_v1.seqVectorDif(t1_v2);

    assert(checkIfVectorAreEqual(t1_gpuRv,t1_seqRv));
    std::cout << "vector diff test OK" << std::endl;
}

void test2()
{
    gpuLinAlg::Matrix t2_matrix{ROWS,COLUMNS};
    t2_matrix.randomInit(1,10);

    gpuLinAlg::Vector t2_v1{COLUMNS};
    t2_v1.randomInit(1,10);

    gpuLinAlg::Vector t2_gpuRv = t2_matrix.gpuMatrixVectorMult(t2_v1);
    gpuLinAlg::Vector t2_seqRv = t2_matrix.seqMatrixVectorMult(t2_v1);

    assert(checkIfVectorAreEqual(t2_gpuRv,t2_seqRv));
    std::cout << "Matrix vector multiplication test OK" << std::endl;
}

void test3()
{
    gpuLinAlg::Matrix a{5,5};
    a.randomInit(0,1);

    printMatrix(a);
    gpuLinAlg::CSRMatrix b = a.toCSRMatrix();
    std::cout << b ;
}

void test4()
{
    gpuLinAlg::CSRMatrix a {3,5,4};
    a.randomInit(1,5);

    gpuLinAlg::Matrix b = a.toMatrix();

    std::cout << a;
    printMatrix(b);

    gpuLinAlg::Matrix{2,2}.data();

}

void test5()
{
    gpuLinAlg::CSRMatrix a{3,5,4};
    a.randomInit(2,4);

    gpuLinAlg::Vector x{5};
    x.valInit(2);

    gpuLinAlg::Vector c = a.seqMatrixVectorMult(x);

    std::cout << a;
    printVector(x);
    printVector(c);
}

void test6()
{
    gpuLinAlg::CSRMatrix a{80000,50000,10000000};
    a.randomInit(2,3);

    gpuLinAlg::Vector x{50000};
    x.valInit(1);

    gpuLinAlg::Vector c = a.seqMatrixVectorMult(x);
    gpuLinAlg::Vector d = a.gpuMatrixVectorMult(x);

    assert(checkIfVectorAreEqual(c,d));
    std::cout << "CSRMatrix vector multiplication test OK" << std::endl;

}

int main(void)
{
/*    
    test1();
    std::cout << "=======================================" << std::endl;
    test2();
    std::cout << "=======================================" << std::endl;
    test3();
    std::cout << "=======================================" << std::endl;
    test4();
    std::cout << "=======================================" << std::endl;
    test5();
    std::cout << "=======================================" << std::endl;
    test6();
*/
    return 0;
}