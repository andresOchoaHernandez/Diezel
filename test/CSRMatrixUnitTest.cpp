#include <cassert>
#include <iostream>

#include "LinearAlgebra.hpp"
#include "MeasureTime.hpp"

void test_csrMatrixWithEmptyRows()
{
    // REPEAT ROW INDEX FOR EMPTY ROWS LEADS TO CORRECT BEHAVIOUR

    using LinearAlgebra::CSRMatrix;
    using LinearAlgebra::Vector;

    CSRMatrix a{3,3,4};
    unsigned* rowsVec = a.getRowsArray();
    unsigned* colsVec = a.getColsArray();
    double*   valsVec = a.getValsArray();

    rowsVec[0] = 0; rowsVec[1] = 1; rowsVec[2] = 1; rowsVec[3] = 4;
    colsVec[0] = 2; colsVec[1] = 0; colsVec[2] = 1; colsVec[3] = 2;
    valsVec[0] = 1.0; valsVec[1] = 3.0; valsVec[2] = 1.0; valsVec[3] = 4.0;

    std::cout << a;

    Vector x{3};
    x[0] = 1; x[1] = 2; x[2] = 3;

    Vector r1 = a.matrixVectorMult(x);

    std::cout << r1;

    CSRMatrix b{2,3,4};
    unsigned* r = b.getRowsArray();
    unsigned* c = b.getColsArray();
    double*   v = b.getValsArray();

    r[0] = 0; r[1] = 1; r[2] = 4;
    c[0] = 2; c[1] = 0; c[2] = 1; c[3] = 2;
    v[0] = 1.0; v[1] = 3.0; v[2] = 1.0; v[3] = 4.0;

    std::cout << b;

    Vector r2 = b.matrixVectorMult(x);

    std::cout << r2;
}

void test_another_test()
{
    using LinearAlgebra::CSRMatrix;
    using LinearAlgebra::CSCMatrix;
    using LinearAlgebra::Matrix;

    Matrix a{3,5};
    a.randomInit(0,1);

    std::cout << "Matrix explicit form: " << std::endl << a; 

    CSCMatrix b = a.toCSCMatrix();

    std::cout << "Matrix CSC form: " << std::endl << b;

    CSRMatrix f = b.toCSR();

    std::cout << "Matrix CSR form: " << std::endl << f;
}

void test_CSCToCSR()
{
    using LinearAlgebra::CSRMatrix;
    using LinearAlgebra::CSCMatrix;
    using LinearAlgebra::Matrix;

    Matrix a{3u,5u};

    a.valInit(0);

    a[2] = 1; a[8] = 3;
    a[3] = 2; a[9] = 4;
    a[4] = 3; a[10] = 2;
    a[5] = 1; a[11] = 1;
    a[7] = 5; a[13] = 1;

    std::cout << "Explicit format: " << std::endl << a;

    CSCMatrix b = a.toCSCMatrix();

    std::cout << "CSC format: " << std::endl << b;

    CSRMatrix res = b.toCSR();
    std::cout << "CSR format: " << std::endl << res;
}

void test_matrixVectorMultSpeedUp()
{
    using LinearAlgebra::Vector;
    using LinearAlgebra::CSRMatrix;
    using LinearAlgebra::Matrix;
    using MeasureTime::Timer;

    Timer t;

    const unsigned rows    = 10000;
    const unsigned columns = 30000; 


    Matrix a{rows,columns};
    a.randomInit(0,1);

    Vector x{columns};
    x.randomInit(100,200);

    t.begin();
    Vector r1 = a.matrixVectorMult(x);
    t.end("[Matrix]Sequential matrix vector multiplication");

    CSRMatrix f = a.toCSRMatrix();

    t.begin();
    Vector r2 = f.matrixVectorMult(x);
    t.end("[CSRMatrix]Sequential matrix vector multiplication");

    t.begin();
    Vector r3 = f.gpu_matrixVectorMult(x);
    t.end("[CSRMatrix]GPU matrix vector multiplication");

    assert(r1 == r2 && r2 == r3);

}

void test_matrixVectorMult()
{
    /*
    | 0 0 1 2 3 |
    | 1 0 5 3 4 |
    | 2 1 0 1 0 |

    rows | 0 3 7 10 
    cols | 2 3 4 0 2 3 4 0 1 3 
    vals | 1 2 3 1 5 3 4 2 1 1 
    */
    using LinearAlgebra::Vector;
    using LinearAlgebra::CSRMatrix;
    using LinearAlgebra::Matrix;

    Matrix a{3u,5u};

    a.valInit(0);

    a[2] = 1; a[8] = 3;
    a[3] = 2; a[9] = 4;
    a[4] = 3; a[10] = 2;
    a[5] = 1; a[11] = 1;
    a[7] = 5; a[13] = 1;

    CSRMatrix b = a.toCSRMatrix();

    Vector x{5u};
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;
    x[3] = 4;
    x[4] = 5;

    Vector r = b.matrixVectorMult(x);

    Vector r1 = b.gpu_matrixVectorMult(x);

    assert(r == r1);
}

int main()
{
    //test_matrixVectorMult();
    //test_matrixVectorMultSpeedUp();
    //test_CSCToCSR();
    //test_another_test();

    test_csrMatrixWithEmptyRows();

    return 0;
}