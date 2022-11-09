#include <random>
#include <chrono>
#include <iostream>
#include <vector>

#include "LinearAlgebra.hpp"
#include "CSRMatrixKernels.cu"

namespace LinearAlgebra
{
    CSRMatrix::CSRMatrix(unsigned nRows,unsigned nCols, unsigned nNzElems):
    _nRows{nRows},
    _nCols{nCols},
    _nNzElems{nNzElems},
    _rows{new unsigned[_nRows + 1u]},
    _cols{new unsigned[_nNzElems]},
    _vals{new int[_nNzElems]}
    {
        if((nNzElems < _nRows) || ((_nRows * _nCols) < _nNzElems)) throw std::runtime_error{"Non Zero elements must be at least equal to Rows but less than rows x cols"};
    }

    CSRMatrix::CSRMatrix(const CSRMatrix& matrix):
    _nRows{matrix._nRows},
    _nCols{matrix._nCols},
    _nNzElems{matrix._nNzElems},
    _rows{new unsigned[_nRows + 1u]},
    _cols{new unsigned[_nNzElems]},
    _vals{new int[_nNzElems]}
    {
        for(unsigned i = 0u ; i <= _nRows ; i++)
        {
            _rows[i] = matrix._rows[i];
        }

        for(unsigned i = 0u; i < _nNzElems; i++)
        {
            _cols[i] = matrix._cols[i];
            _vals[i] = matrix._vals[i];
        }
    
    }
    CSRMatrix::CSRMatrix(CSRMatrix&& mat)
    {
        _nRows    = mat._nRows;
        _nCols    = mat._nCols;
        _nNzElems = mat._nNzElems; 

        _rows = mat._rows;
        _cols = mat._cols;
        _vals = mat._vals;

        mat._nRows    = 0u;
        mat._nCols    = 0u;
        mat._nNzElems = 0u;

        mat._rows = nullptr; 
        mat._cols = nullptr;
        mat._vals = nullptr;
    }
    CSRMatrix::~CSRMatrix(){delete[] _rows;delete[] _cols;delete[] _vals;}

    Vector CSRMatrix::gpuMatrixVectorMult(const Vector& v1) const
    {
        if(_nCols != v1.len()) throw std::runtime_error{"Matrix dimensions and vector dimensions don't match"};

        Vector rv{_nRows};

        int* rows_device;
        int* cols_device;
        int* vals_device; 
        
        int* v1_device; 
        int* rv_device;

        cudaMalloc(&rows_device,sizeof(int)*(_nRows + 1));
        cudaMalloc(&cols_device,sizeof(int)*_nNzElems);
        cudaMalloc(&vals_device,sizeof(int)*_nNzElems);

        cudaMalloc(&v1_device,sizeof(int)*v1.len());
        cudaMalloc(&rv_device,sizeof(int)*rv.len());

        cudaMemcpy(rows_device,_rows,sizeof(int)*(_nRows + 1),cudaMemcpyHostToDevice);
        cudaMemcpy(cols_device,_cols,sizeof(int)*_nNzElems,cudaMemcpyHostToDevice);
        cudaMemcpy(vals_device,_vals,sizeof(int)*_nNzElems,cudaMemcpyHostToDevice);

        cudaMemcpy(v1_device,&v1[0u],sizeof(int)*v1.len(),cudaMemcpyHostToDevice);

        const unsigned threadsPerBlock = 1024u;
        const unsigned numberOfBlocks = _nRows < threadsPerBlock? 1u: (_nRows % threadsPerBlock == 0u? _nRows/threadsPerBlock:_nRows/threadsPerBlock +1u);
        dim3 dimGrid(numberOfBlocks,1,1);
        dim3 dimBlock(threadsPerBlock,1,1);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        csrMatrixVectorMultKernel<<<dimGrid,dimBlock>>>(rows_device,cols_device,vals_device,v1_device,rv_device,_nRows);
        cudaDeviceSynchronize();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Cuda kernel for csr matrix vector multiplication took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

        cudaMemcpy(&rv[0u],rv_device,sizeof(int)*rv.len(),cudaMemcpyDeviceToHost);

        cudaFree(rows_device);
        cudaFree(cols_device);
        cudaFree(vals_device);

        cudaFree(v1_device);
        cudaFree(rv_device);

        cudaDeviceReset();

        return rv;
    }
    Vector CSRMatrix::seqMatrixVectorMult(const Vector& v1) const
    {
        if(_nCols != v1.len()) throw std::runtime_error("Matrix and Vector's dimensions don't match!");

        Vector rv{_nRows};

        unsigned startRow = 0u; 
        unsigned endRow   = 0u;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for(unsigned i = 0u; i < _nRows ; i++ )
        {
            startRow = _rows[i];
            endRow   = _rows[i + 1u];

            rv[i] = 0;

            for(unsigned j = startRow; j < endRow; j++)
            {
                rv[i] += _vals[j] * v1[_cols[j]];
            } 
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Sequential csr matrix vector multiplication took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;


        return rv;
    }
    void CSRMatrix::randomInit(int a,int b)
    {
        if(a == 0 || b == 0) throw std::runtime_error("a and b must be != 0");

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> vals_dist(a,b);
        std::uniform_int_distribution<std::mt19937::result_type> cols_dist(0,_nCols-1);

        std::vector<char> matrixIndexes(_nRows * _nCols,'0');

        for(unsigned i = 0u; i < _nRows ; i ++)
        {
            matrixIndexes[i*_nCols + cols_dist(rng)] = '1';
        }

        unsigned elemsToDistribute = _nNzElems - _nRows;
        unsigned randomIndex;
        bool rowIsFull = false;

        while(elemsToDistribute > 0u)
        {
            for(unsigned i = 0u ; i < _nRows; i++)
            {
                if(elemsToDistribute == 0u) break;

                randomIndex = cols_dist(rng) ;

                if(matrixIndexes[i*_nCols + randomIndex] == '1')
                {
                    for(unsigned j = 0u ; j < _nCols; j++ )
                    {
                        if(matrixIndexes[i*_nCols + j] == '0')
                        {
                            rowIsFull = false;
                            matrixIndexes[i*_nCols + j] = '1';
                            elemsToDistribute--;
                            break;
                        }
                        rowIsFull = true;
                    }

                    if(rowIsFull) continue;
                    
                }
                else
                {
                    matrixIndexes[i*_nCols + randomIndex] = '1';
                    elemsToDistribute--;
                }
            }
        }

        unsigned NzElemsIndex = 0u;

        _rows[0u] = 0u;

        for(unsigned i = 0u ; i < _nRows ; i++)
        {
            for(unsigned j = 0u ; j < _nCols ; j++)
            {
                if(matrixIndexes[i*_nCols + j] == '1')
                {
                    _cols[NzElemsIndex] = j; 
                    _vals[NzElemsIndex] = vals_dist(rng);
                    NzElemsIndex++; 
                }
            }
            _rows[i + 1u] = NzElemsIndex;
        }
    }

    Matrix CSRMatrix::toMatrix()
    {
        Matrix result{_nRows,_nCols};
        result.valInit(0);

        unsigned startRow,endRow;

        for(unsigned i = 0u ; i < _nRows; i++)
        {
            startRow = _rows[i];
            endRow   = _rows[i+1u];
            for(unsigned j = startRow; j < endRow ; j++)
            {
                result[i*_nCols + _cols[j]] = _vals[j];
            }
        }

        return result;
    }

    unsigned  CSRMatrix::rows()const{return _nRows;}
    unsigned  CSRMatrix::cols()const{return _nCols;}
    unsigned  CSRMatrix::nonZeroElements()const{return _nNzElems;}
    unsigned* CSRMatrix::getRowsArray(){return _rows;}
    unsigned* CSRMatrix::getColsArray(){return _cols;}
    int*      CSRMatrix::getValsArray(){return _vals;}

    std::ostream& operator<<(std::ostream& stream, const CSRMatrix& operand)
    {
        stream << "rows | ";

        for(unsigned i = 0u; i <= operand._nRows; i++)
        {
            stream << operand._rows[i] << " ";
        }

        stream << std::endl << "cols | ";

        for(unsigned i = 0u; i < operand._nNzElems; i++)
        {
            stream << operand._cols[i] << " ";        
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