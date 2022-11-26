__global__ void matrixVectorMultKernel(const double* matrix, const double* v1, double* rv, const unsigned rows, const unsigned cols)
{
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= rows) return;

    double acc = 0;

    for(unsigned i = 0u; i < cols; i++ )
    {
        acc+= matrix[row * cols + i] * v1[i];
    }    

    rv[row] = acc;
}