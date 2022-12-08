__global__ void csrMatrixVectorMultKernel(const unsigned* csrRows, const unsigned* csrCols, const float*csrVals, const float* v1, float* rv,const unsigned rows)
{
    const unsigned globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned rowIndex    = globalIndex / 32u;

    if(rowIndex >= rows) return;

    const unsigned startRow = csrRows[rowIndex];
    const unsigned endRow   = csrRows[rowIndex + 1];

    const unsigned elementsOnTheRow = endRow - startRow;

    __shared__ float vals[32];

    for(unsigned offset = 0; offset < (unsigned)ceilf(elementsOnTheRow/32.0f) ; offset++)
    {
        vals[threadIdx.x] = 0.0f;

        if(threadIdx.x < (elementsOnTheRow - 32u*offset))
        {
            vals[threadIdx.x] = csrVals[startRow + 32u*offset + threadIdx.x] * v1[csrCols[startRow + 32u*offset + threadIdx.x]];
        }
        __syncthreads();

        for(unsigned i = 1u; i < 32u ; i*=2u)
        {
            const unsigned index = threadIdx.x*i*2;

            if(index < 32)
                vals[index] += vals[index + i]; 
            __syncthreads();
        }   

        if(threadIdx.x == 0)
        {
            rv[rowIndex] += vals[0];
        }
    }
}