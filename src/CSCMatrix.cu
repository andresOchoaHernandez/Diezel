#include "LinearAlgebra.hpp"
#include "CSCMatrixKernels.cu"

namespace LinearAlgebra
{
    CSCMatrix::CSCMatrix(unsigned nRows,unsigned nCols, unsigned nNzElems):
    _nRows{nRows},
    _nCols{nCols},
    _nNzElems{nNzElems},
    _cols{new unsigned[_nCols + 1u]},
    _rows{new unsigned[_nNzElems]},
    _vals{new int[_nNzElems]}
    {
        if((nNzElems < _nCols) || ((_nRows * _nCols) < _nNzElems)) throw std::runtime_error{"Non Zero elements must be at least equal to Cols but less than rows x cols"};
    }
    CSCMatrix::CSCMatrix(const CSCMatrix& matrix):
    _nRows{matrix._nRows},
    _nCols{matrix._nCols},
    _nNzElems{matrix._nNzElems},
    _cols{new unsigned[_nCols + 1u]},
    _rows{new unsigned[_nNzElems]},
    _vals{new int[_nNzElems]}
    {
        for(unsigned i = 0u ; i <= _nCols ; i++)
        {
            _cols[i] = matrix._cols[i];
        }

        for(unsigned i = 0u ; i < _nNzElems ; i++)
        {
            _rows[i] = matrix._rows[i];
            _vals[i] = matrix._vals[i];
        }
    }
    CSCMatrix::CSCMatrix(CSCMatrix&& mat)
    {
        _nRows    = mat._nRows;
        _nCols    = mat._nCols;
        _nNzElems = mat._nNzElems; 

        _cols = mat._cols;
        _rows = mat._rows;
        _vals = mat._vals;

        mat._nRows    = 0u;
        mat._nCols    = 0u;
        mat._nNzElems = 0u;

        mat._cols = nullptr;
        mat._rows = nullptr; 
        mat._vals = nullptr;
    }
    CSCMatrix::~CSCMatrix(){delete[] _cols;delete[] _rows;delete[] _vals;}

    void CSCMatrix::randomInit(int a,int b)
    {
        // TODO:
    }

    Vector CSCMatrix::matrixVectorMult(const Vector& v1)const
    {
        if(_nCols != v1.len()) throw std::runtime_error("Matrix and Vector's dimensions don't match!");
        
        Vector result{_nRows};

        result.valInit(0);

        for(unsigned i = 0u ; i < _nCols ; i++)
        {
            unsigned startColumn = _cols[i];
            unsigned endColumn   = _cols[i + 1u];

            for(unsigned j = startColumn ; j < endColumn ; j++)
            {
                result[_rows[j]] += _vals[j] * v1[i];
            }
        }

        return result;
    }

    Vector CSCMatrix::gpu_matrixVectorMult(const Vector& v1)const
    {
        // TODO:
        return {1};
    }

    Matrix CSCMatrix::toMatrix() const 
    {
        // TODO:
        return {1,1};
    }

    CSRMatrix CSCMatrix::toCSR() const
    {
        //TODO: Urgently

        return {1,1,1};
    }

    unsigned  CSCMatrix::rows()const
    {
        return _nRows;
    }
    unsigned  CSCMatrix::cols()const
    {
        return _nCols;
    }
    unsigned  CSCMatrix::nonZeroElements()const
    {
        return _nNzElems;
    }
    unsigned* CSCMatrix::getColsArray()
    {
        return _cols;
    }
    unsigned* CSCMatrix::getRowsArray()
    {
        return _rows;
    }
    int*      CSCMatrix::getValsArray()
    {
        return _vals;
    }

    std::ostream& operator<<(std::ostream& stream, const CSCMatrix& operand)
    {
        stream << "cols | ";

        for(unsigned i = 0u; i <= operand._nCols; i++)
        {
            stream << operand._cols[i] << " ";
        }

        stream << std::endl << "rows | ";

        for(unsigned i = 0u; i < operand._nNzElems; i++)
        {
            stream << operand._rows[i] << " ";        
        }

        stream << std::endl << "vals | ";

        for(unsigned i = 0u; i < operand._nNzElems; i++)
        {
            stream << operand._vals[i] << " ";        
        }

        stream << std::endl;

        return stream;
    }
}