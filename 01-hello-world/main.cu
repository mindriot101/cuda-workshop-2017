#include <stdio.h>

__global__ void hello_world(float f) {
    int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y;
    int thread_idx = block_idx * blockDim.y * blockDim.x + threadIdx.x + threadIdx.y * blockDim.x;

    printf("Hello block %i (x %i, y %i, z %i) running thread %i (x %i, y %i, z %i), f=%f\n",
            block_idx, blockIdx.x, blockIdx.y, blockIdx.z,
            thread_idx, threadIdx.x, threadIdx.y, threadIdx.z,
            f);
}

int main() {
    dim3 grid(2, 2, 2);
    dim3 block(2, 2, 1);

    hello_world<<<grid, block>>>(1.2345f);
    cudaDeviceReset();
    return 0;
}
