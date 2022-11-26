#include <random>
#include <chrono>
#include <iostream>

#include "LinearAlgebra.hpp"
#include "VectorKernels.cu"

namespace LinearAlgebra
{
    Vector::Vector(unsigned len):_len{len},_vec{new double[_len]}{}
    Vector::Vector(const Vector& vector):_len{vector._len},_vec{new double[_len]}
    {
        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++)
        {
            _vec[i] = vector._vec[i];
        }
    }
    Vector::Vector(Vector&& v)
    {
        _len = v._len;
        _vec = v._vec;
        v._len = 0u;
        v._vec = nullptr;
    }
    Vector::~Vector(){delete[] _vec;}


    Vector Vector::operator+(const Vector& other)const
    {
        if( _len != other.len()) throw std::runtime_error{"Vectors dimensions don't match"};

        Vector result{_len};
        
        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i] + other[i];
        }

        return result;
    }

    Vector Vector::operator-(const Vector& other)const
    {
        if( _len != other.len()) throw std::runtime_error{"Vectors dimensions don't match"};
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i]- other[i];
        }

        return result;
    }

    Vector Vector::operator*(const Vector& other)const
    {
        if( _len != other.len()) throw std::runtime_error{"Vectors dimensions don't match"};
        
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i]*other[i];
        }

        return result;
    }

    Vector Vector::operator/(const Vector& other)const
    {
        if( _len != other.len()) throw std::runtime_error{"Vectors dimensions don't match"};
        
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i] / other[i];
        }

        return result;
    }

    Vector Vector::operator+(const double constant)const
    {   
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i] + constant;
        }

        return result;
    }

    Vector Vector::operator-(const double constant)const
    {
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i] - constant;
        }

        return result;
    }

    Vector Vector::operator*(const double constant)const
    {
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i] * constant;
        }

        return result;
    }

    Vector Vector::operator/(const double constant)const
    {
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i] / constant;
        }

        return result;
    }


    Vector Vector::gpu_diff(const Vector& v2)const
    {
        if( _len != v2.len()) throw std::runtime_error{"Vectors dimensions don't match"};

        Vector rv{_len};

        double *v1_device;double *v2_device;double *rv_device;

        cudaMalloc(&v1_device,sizeof(double)*_len);
        cudaMalloc(&v2_device,sizeof(double)*v2.len());
        cudaMalloc(&rv_device,sizeof(double)*rv.len());

        cudaMemcpy(v1_device,_vec,sizeof(double)*_len,cudaMemcpyHostToDevice);
        cudaMemcpy(v2_device,&v2[0u],sizeof(double)*v2.len(),cudaMemcpyHostToDevice);

        const unsigned threadsPerBlock = 1024u;
        const unsigned numberOfBlocks = _len < threadsPerBlock? 1u: (_len % threadsPerBlock == 0u? _len/threadsPerBlock:_len/threadsPerBlock +1u);
        dim3 dimGrid(numberOfBlocks,1,1);
        dim3 dimBlock(threadsPerBlock,1,1);
        
        vectorDifKernel<<<dimGrid,dimBlock>>>(v1_device,v2_device,rv_device,_len);
        cudaDeviceSynchronize();

        cudaMemcpy(&rv[0u],rv_device,sizeof(double)*rv.len(),cudaMemcpyDeviceToHost);

        cudaFree(v1_device);
        cudaFree(v2_device);
        cudaFree(rv_device);

        cudaDeviceReset();

        return rv;
    }

    Vector Vector::gpu_sum(const Vector& v2)const
    {
        //TODO:
        return v2;
    }

    void Vector::randomInit(double a, double b)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(a,b);

        #pragma omp parallel for
        for (unsigned i = 0u ; i < _len ; i++ )
            _vec[i] = dist(rng);
    }

    void Vector::valInit(double val)
    {
        #pragma omp parallel for
        for (unsigned i = 0u ; i < _len ; i++ )
            _vec[i] = val;
    }

    unsigned Vector::len()const{ return _len; }
    double* Vector::getVec(){ return _vec; }

    double& Vector::operator [](unsigned i){return _vec[i];}
    const double& Vector::operator [](unsigned i)const{return _vec[i];}

    bool Vector::operator==(const Vector& other) const
    {
        if(_len != other.len()) return false;

        for(unsigned i = 0u ; i < _len ; i++)
        {
            if(_vec[i] != other[i]) return false;
        }

        return true;
    }

    std::ostream& operator<<(std::ostream& stream, const Vector& operand)
    {
        for(unsigned i = 0u ; i < operand._len ; i++)
            stream << operand[i] << " ";
    
        stream << std::endl;

        return stream;
    }
}