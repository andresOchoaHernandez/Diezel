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