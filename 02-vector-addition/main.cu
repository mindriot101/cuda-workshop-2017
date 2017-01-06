#include <stdio.h>
#include <cuda.h>
#include "cuda_helpers.h"
#define N 4096

__global__ void vector_add_kernel(int *d_a, int *d_b, int *d_c) {
    int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    int thread_idx = block_idx * blockDim.y * blockDim.x + threadIdx.x + threadIdx.y * blockDim.x;

    if (thread_idx < N) {

        printf("Hello block %i (x %i, y %i, z %i) running thread %i (x %i, y %i, z %i)\n",
                block_idx, blockIdx.x, blockIdx.y, blockIdx.z,
                thread_idx, threadIdx.x, threadIdx.y, threadIdx.z);

        d_c[thread_idx] = d_a[thread_idx] + d_b[thread_idx];
    }
}

int main() {
    struct cudaDeviceProp props;
    int status = cudaGetDeviceProperties(&props, 0);

    printf("Threads per block: (%i %i %i)\n",
            props.maxThreadsDim[0],
            props.maxThreadsDim[1],
            props.maxThreadsDim[2]);

    dim3 grid(4, 8);
    dim3 block(128);

    int h_a[N];
    int h_b[N];
    int h_c[N];
    int *d_a, *d_b, *d_c;

    for (int i=0; i<N; i++) {
        h_a[i] = i;
        h_b[i] = i + 1;
    }

    CUDA_CALL(cudaMalloc(&d_a, N * sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_b, N * sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_c, N * sizeof(int)));

    CUDA_CALL(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime;

    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start, 0));

    vector_add_kernel<<<grid, block>>>(d_a, d_b, d_c);
    cuda_peek();

    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

    CUDA_CALL(cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i=0; i<N; i++) {
        printf("%i+%i = %i\n", h_a[i], h_b[i], h_c[i]);
    }

    printf("Time taken: %f ms\n", elapsedTime);

    CUDA_CALL(cudaFree(d_a));
    CUDA_CALL(cudaFree(d_b));
    CUDA_CALL(cudaFree(d_c));

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    CUDA_CALL(cudaDeviceReset());
    return 0;
}
