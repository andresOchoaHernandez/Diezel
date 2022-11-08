#include <iostream>

#include "gpuLinAlg.hpp"

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

void printMatrix(gpuLinAlg::Matrix& m)
{
    std::cout << "---------------------" << std::endl;

    for(unsigned i = 0u; i < m.rows() ; i++)
    {
        for(unsigned j = 0u ; j < m.cols(); j++)
        {
            std::cout << m[i*m.cols() + j] << " ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl << "---------------------" << std::endl;
}

void printVector(gpuLinAlg::Vector& v)
{
    std::cout << "---------------------" << std::endl;

    for(unsigned i = 0u; i < v.len() ; i++)
        std::cout << v[i] <<" ";

    std::cout << std::endl << "---------------------" << std::endl;
}