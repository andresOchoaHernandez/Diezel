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