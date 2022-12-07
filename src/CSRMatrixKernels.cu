__global__ void csrMatrixVectorMultKernel(const unsigned* csrRows, const unsigned* csrCols, const float*csrVals, const float* v1, float* rv,const unsigned rows)
{
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) return;

    const unsigned rowStart = csrRows[row]; 
    const unsigned rowEnd   = csrRows[row + 1];

    float acc = 0.0f;

    for(unsigned i = rowStart ; i < rowEnd ; i++ )
    {
        acc += csrVals[i] * v1[csrCols[i]]; //TODO: use an acc to limit global access memory
    }

    rv[row] = acc;
}