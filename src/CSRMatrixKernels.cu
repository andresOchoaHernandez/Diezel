__global__ void csrMatrixVectorMultKernel(const int* csrRows, const int* csrCols, const int*csrVals, const int* v1, int* rv,const unsigned rows)
{
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) return;

    const unsigned rowStart = csrRows[row]; 
    const unsigned rowEnd   = csrRows[row + 1];

    rv[row] = 0;

    for(unsigned i = rowStart ; i < rowEnd ; i++ )
    {
        rv[row] += csrVals[i] * v1[csrCols[i]];
    }
}