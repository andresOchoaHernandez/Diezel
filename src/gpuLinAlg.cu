#include <random>
#include <chrono>
#include <iostream>
#include <vector>

#include "gpuLinAlg.hpp"
#include "kernels.cu"

namespace gpuLinAlg
{
    Vector::Vector(unsigned len):_len{len},_vec{new int[_len]}{}
    Vector::Vector(const Vector& vector):_len{vector._len},_vec{new int[_len]}{for(unsigned i = 0u ; i < _len ; i++)_vec[i] = vector._vec[i];}
    Vector::Vector(Vector&& v)
    {
        _len = v._len;
        _vec = v._vec;
        v._len = 0u;
        v._vec = nullptr;
    }
    Vector::~Vector(){delete[] _vec;}

    Vector Vector::gpuVectorDif(const Vector& v2)const
    {
        if( _len != v2.len()) throw std::runtime_error{"Vectors dimensions don't match"};

        Vector rv{_len};

        int *v1_device;int *v2_device;int *rv_device;

        cudaMalloc(&v1_device,sizeof(int)*_len);
        cudaMalloc(&v2_device,sizeof(int)*v2.len());
        cudaMalloc(&rv_device,sizeof(int)*rv.len());

        cudaMemcpy(v1_device,_vec,sizeof(int)*_len,cudaMemcpyHostToDevice);
        cudaMemcpy(v2_device,&v2[0u],sizeof(int)*v2.len(),cudaMemcpyHostToDevice);

        const unsigned threadsPerBlock = 1024u;
        const unsigned numberOfBlocks = _len < threadsPerBlock? 1u: (_len % threadsPerBlock == 0u? _len/threadsPerBlock:_len/threadsPerBlock +1u);
        dim3 dimGrid(numberOfBlocks,1,1);
        dim3 dimBlock(threadsPerBlock,1,1);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        vectorDifKernel<<<dimGrid,dimBlock>>>(v1_device,v2_device,rv_device,_len);
        cudaDeviceSynchronize();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Cuda kernel for vector diff took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

        cudaMemcpy(&rv[0u],rv_device,sizeof(int)*rv.len(),cudaMemcpyDeviceToHost);

        cudaFree(v1_device);
        cudaFree(v2_device);
        cudaFree(rv_device);

        cudaDeviceReset();

        return rv;
    }

    Vector Vector::seqVectorDif(const Vector& v2) const
    {
        if( _len != v2.len()) throw std::runtime_error{"Vectors dimensions don't match"};
        
        Vector rv{_len};

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for(unsigned i = 0u ; i < rv.len(); i++)
        {
            rv[i] = _vec[i] - v2[i];
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Sequential vector diff took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
        
        return rv;
    }

    void Vector::randomInit(int a, int b)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(a,b);

        for (unsigned i = 0u ; i < _len ; i++ )
            _vec[i] = dist(rng);
    }

    void Vector::valInit(int val)
    {
        for (unsigned i = 0u ; i < _len ; i++ )
            _vec[i] = val;
    }

    unsigned Vector::len()const{ return _len; }
    int* Vector::getVec(){ return _vec; }

    int& Vector::operator [](unsigned i){return _vec[i];}
    const int& Vector::operator [](unsigned i)const{return _vec[i];}

    std::ostream& operator<<(std::ostream& stream, const Vector& operand)
    {
        for(unsigned i = 0u ; i < operand._len ; i++)
            stream << operand[i] << " ";
    
        stream << std::endl;

        return stream;
    }


/*========================================================================================================================================*/

    Matrix::Matrix(unsigned rows, unsigned cols):_rows{rows},_cols{cols},_data{new int[_rows*_cols]}{}
    Matrix::Matrix(const Matrix& matrix):_rows{matrix._rows},_cols{matrix._cols},_data{new int[_rows*_cols]}
    {
        for(unsigned i = 0u; i < _rows; i++)
        {
            for(unsigned j = 0u; j < _cols; j++)
            {
                _data[i*_cols + j] = matrix._data[i*_cols + j]; 
            }
        }
    }
    Matrix::Matrix(Matrix&& mat)
    {
        _rows = mat._rows;
        _cols = mat._cols;
        _data = mat._data;

        mat._rows = 0u;
        mat._cols = 0u;
        mat._data = nullptr;
    }
    Matrix::~Matrix(){delete[]_data;}

    Vector Matrix::gpuMatrixVectorMult(const Vector& v1) const
    {
        if(_cols != v1.len()) throw std::runtime_error{"Matrix dimensions and vector dimensions don't match"};

        Vector rv{_rows};

        int* matrix_device; int* v1_device; int* rv_device;

        cudaMalloc(&matrix_device,sizeof(int)*_rows*_cols);
        cudaMalloc(&v1_device,sizeof(int)*v1.len());
        cudaMalloc(&rv_device,sizeof(int)*rv.len());

        cudaMemcpy(matrix_device,_data,sizeof(int)*_rows*_cols,cudaMemcpyHostToDevice);
        cudaMemcpy(v1_device,&v1[0u],sizeof(int)*v1.len(),cudaMemcpyHostToDevice);

        const unsigned threadsPerBlock = 1024u;
        const unsigned numberOfBlocks = _rows < threadsPerBlock? 1u: (_rows % threadsPerBlock == 0u? _rows/threadsPerBlock:_rows/threadsPerBlock +1u);
        dim3 dimGrid(numberOfBlocks,1,1);
        dim3 dimBlock(threadsPerBlock,1,1);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        matrixVectorMultKernel<<<dimGrid,dimBlock>>>(matrix_device,v1_device,rv_device,_rows,_cols);
        cudaDeviceSynchronize();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Cuda kernel for matrix vector multiplication took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

        cudaMemcpy(&rv[0u],rv_device,sizeof(int)*rv.len(),cudaMemcpyDeviceToHost);

        cudaFree(matrix_device);
        cudaFree(v1_device);
        cudaFree(rv_device);

        cudaDeviceReset();

        return rv;
    }
    Vector Matrix::seqMatrixVectorMult(const Vector& v1) const
    {
        if(_cols != v1.len()) throw std::runtime_error{"Matrix dimensions and vector dimensions don't match"};

        Vector rv{_rows};

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for( unsigned i = 0u; i < _rows; i++ )
        {
            rv[i] = 0;

            for( unsigned j = 0u; j < _cols; j++ )
            {
                rv[i] += _data[i*_cols + j] * v1[j]; 
            }
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Sequential matrix vector multiplication took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

        return rv;
    }
    void Matrix::randomInit(int a,int b)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(a,b);

        for(unsigned i = 0u; i < _rows; i++)
        {
            for(unsigned j = 0u; j < _cols; j++)
            {
                _data[i*_cols + j] = dist(rng); 
            }
        }
    }
    
    void Matrix::valInit(int val)
    {
        for (unsigned i = 0u ; i <  _rows ; i++ )
        {
            for(unsigned j = 0u ;  j < _cols ; j++ )
            {
                _data[i*_cols + j] = val;
            }
        }
    }

    CSRMatrix Matrix::toCSRMatrix() const
    {
        unsigned nonZeroElements = 0u;
        bool emptyRow = true;

        for(unsigned i = 0u ; i < _rows ; i++)
        {
            for(unsigned j = 0u ; j < _cols ; j++)
            {
                if(_data[i*_cols + j] != 0)
                {
                    nonZeroElements++;
                    emptyRow = false;
                }
            }
            if(emptyRow) throw std::runtime_error{"CSRMatrix doesn't allow empty rows"};
        }

        CSRMatrix csrMatrix{_rows,_cols,nonZeroElements};

        unsigned* rowsVec = csrMatrix.getRowsArray();
        unsigned* colsVec = csrMatrix.getColsArray();
        int*      valsVec = csrMatrix.getValsArray();

        unsigned NzElemsIndex = 0u;

        rowsVec[0u] = 0u;

        for(unsigned i = 0u ; i < csrMatrix.rows() ; i++)
        {
            for(unsigned j = 0u ; j < csrMatrix.cols() ; j++)
            {
                if(_data[i*csrMatrix.cols() + j] != 0)
                {
                    colsVec[NzElemsIndex] = j; 
                    valsVec[NzElemsIndex] = _data[i*csrMatrix.cols() + j];
                    NzElemsIndex++; 
                }
            }

            rowsVec[i + 1u] = NzElemsIndex;
        }

        return csrMatrix;
    }

    unsigned Matrix::rows()const{return _rows;}
    unsigned Matrix::cols()const{return _cols;}
    int*     Matrix::data(){return _data;}

    int& Matrix::operator [](unsigned i){return _data[i];}
    const int& Matrix::operator [](unsigned i)const{return _data[i];}

    std::ostream& operator<<(std::ostream& stream, const Matrix& operand)
    {
        for(unsigned i = 0u; i < operand._rows ; i++)
        {
            for(unsigned j = 0u; j < operand._cols ; j++)
            {
                stream << operand[i*operand._cols + j] << " ";
            }

            stream << std::endl;
        }

        return stream;
    }

/*========================================================================================================================================*/

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
        //TODO: fix this method

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