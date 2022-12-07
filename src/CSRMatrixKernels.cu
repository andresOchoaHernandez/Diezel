__global__ void csrMatrixVectorMultKernel(const unsigned* csrRows, const unsigned* csrCols, const float*csrVals, const float* v1, float* rv,const unsigned rows)
{
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) return;

    const unsigned rowStart = csrRows[row]; 
    const unsigned rowEnd   = csrRows[row + 1];

    float acc = 0.0f;

    for(unsigned i = rowStart ; i < rowEnd ; i++ )
    {
        acc += csrVals[i] * v1[csrCols[i]];
    }

    rv[row] = acc;
}


// assumes blocks of threads
__global__ void csrMatrixVectorMultKernelReduction(const unsigned* csrRows, const unsigned* csrCols, const float*csrVals, const float* v1, float* rv,const unsigned rows)
{
    const unsigned globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned rowIndex    = globalIndex / 32u;
    const unsigned warpIndex   = globalIndex % 32u;

    if(rowIndex >= rows) return;

    const unsigned startRow = csrRows[rowIndex];
    const unsigned endRow   = csrRows[rowIndex + 1];

    const unsigned elementsOnTheRow = endRow - startRow;

    __shared__ float vals[32];

    //vals[threadIdx.x] = csrVals[]

}