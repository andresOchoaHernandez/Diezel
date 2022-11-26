__global__ void vectorDifKernel(const double* v1, const double* v2, double* rv, const unsigned size)
{
    const unsigned globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(globalIndex >= size) return;

    rv[globalIndex] = v1[globalIndex] - v2[globalIndex];
}