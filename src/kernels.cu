__global__ void vectorDifKernel(const int* v1, const int* v2, int* rv, const unsigned size)
{
    const unsigned globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(globalIndex >= size) return;

    rv[globalIndex] = v1[globalIndex] - v2[globalIndex];
}

__global__ void matrixVectorMultKernel(const int* matrix, const int* v1, int* rv, const unsigned rows, const unsigned cols)
{
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= rows) return;

    int acc = 0;

    for(unsigned i = 0u; i < cols; i++ )
    {
        acc+= matrix[row * cols + i] * v1[i];
    }    

    rv[row] = acc;
}

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