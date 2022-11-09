__global__ void vectorDifKernel(const int* v1, const int* v2, int* rv, const unsigned size)
{
    const unsigned globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(globalIndex >= size) return;

    rv[globalIndex] = v1[globalIndex] - v2[globalIndex];
}