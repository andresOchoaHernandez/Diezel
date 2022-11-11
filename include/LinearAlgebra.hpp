#pragma once

#include <ostream>

namespace LinearAlgebra{

    class Vector; class Matrix; class CSRMatrix;

    class Vector
    {
        unsigned _len;
        
        int*     _vec;

        public:
            Vector(unsigned len);
            Vector(const Vector& vector);
            Vector(Vector&& v);
            ~Vector();

            Vector operator+(const Vector& other)const;
            Vector operator-(const Vector& other)const;
            Vector operator*(const Vector& other)const;
            Vector operator/(const Vector& other)const;

            Vector operator+(const int constant)const;
            Vector operator-(const int constant)const;
            Vector operator*(const int constant)const;
            Vector operator/(const int constant)const;

            Vector gpu_diff(const Vector& v2)const;
            Vector gpu_sum(const Vector& v2)const;

            void randomInit(int a,int b);
            void valInit(int val);

            unsigned len()const;
            int*     getVec();

            int& operator [](unsigned i);
            const int& operator [](unsigned i)const;
            bool operator==(const Vector& other) const;

            friend std::ostream& operator<<(std::ostream& stream, const Vector& operand);
    };

    class Matrix
    {
        unsigned _rows;
        unsigned _cols;

        int* _data;

        public:
            Matrix(unsigned rows, unsigned cols);
            Matrix(const Matrix& matrix);
            Matrix(Matrix&& mat);
            ~Matrix();

            Matrix operator+(const Matrix& other)const;
            Matrix operator-(const Matrix& other)const;
            Matrix operator*(const Matrix& other)const;
            Matrix operator/(const Matrix& other)const;

            Matrix operator+(const int constant)const;
            Matrix operator-(const int constant)const;
            Matrix operator*(const int constant)const;
            Matrix operator/(const int constant)const;

            Vector gpuMatrixVectorMult(const Vector& v1)const;
            Vector seqMatrixVectorMult(const Vector& v1)const;
            void randomInit(int a,int b);
            void valInit(int val);

            CSRMatrix toCSRMatrix() const;

            unsigned rows()const;
            unsigned cols()const;
            int*     data();

            int& operator [](unsigned i);
            const int& operator [](unsigned i)const;
            bool operator==(const Matrix& other) const;

            friend std::ostream& operator<<(std::ostream& stream, const Matrix& operand);
    };

    class CSRMatrix
    {
        unsigned  _nRows;
        unsigned  _nCols;
        unsigned  _nNzElems;

        unsigned *_rows;
        unsigned *_cols;
        int      *_vals;

        public:
            CSRMatrix(unsigned nRows,unsigned nCols, unsigned nNzElems);
            CSRMatrix(const CSRMatrix& matrix);
            CSRMatrix(CSRMatrix&& mat);
            ~CSRMatrix();

            Vector gpuMatrixVectorMult(const Vector& v1)const;
            Vector seqMatrixVectorMult(const Vector& v1)const;
            void randomInit(int a,int b);

            Matrix toMatrix();

            unsigned  rows()const;
            unsigned  cols()const;
            unsigned  nonZeroElements()const;
            unsigned* getRowsArray();
            unsigned* getColsArray();
            int*      getValsArray();

            friend std::ostream& operator<<(std::ostream& stream, const CSRMatrix& operand);
    };
}