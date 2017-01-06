#include <stdio.h>
#include <cuda.h>
#include <omp.h>
#include "cuda_helpers.h"

__global__ void vectorAddKernel(int *a, int *c, int N) {
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tdx < N) {
        c[tdx] = a[tdx] + 1;
    }
}

int main() {
    const int N = 4096000;
    const int NSTREAMS = 2;
    int *a_h[NSTREAMS], *c_h[NSTREAMS];
    int *a_d[NSTREAMS], *b_d[NSTREAMS], *c_d[NSTREAMS];

    cudaStream_t stream[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++) {
        CUDA_CALL(cudaStreamCreate(&stream[i]));

        CUDA_CALL(cudaMallocHost((void **)&a_h[i], (N / 2) * sizeof(int)));
        CUDA_CALL(cudaMallocHost((void **)&c_h[i], (N / 2) * sizeof(int)));

        CUDA_CALL(cudaMalloc((void **)&a_d[i], (N / 2) * sizeof(int)));
        CUDA_CALL(cudaMalloc((void **)&b_d[i], (N / 2) * sizeof(int)));
        CUDA_CALL(cudaMalloc((void **)&c_d[i], (N / 2) * sizeof(int)));
    }

    for (int i = 0; i < NSTREAMS; i++) {
        for (int ii = 0; ii < N / 2; ii++) {
            a_h[i][ii] = i * N / 2 + ii;
        }
    }

    cudaEvent_t start, stop;
    float elapsedTime;

    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start, 0));

    /* CODE */
    dim3 threads(1024);
    dim3 blocks(64, 64);

    CUDA_CALL(cudaMemcpyAsync(a_d[0], a_h[0], (N / 2) * sizeof(int),
                              cudaMemcpyHostToDevice, stream[0]));
    vectorAddKernel<<<blocks, threads, 0, stream[0]>>>(a_d[0], c_d[0], N);
    CUDA_CALL(cudaMemcpyAsync(c_h[0], c_d[0], (N / 2) * sizeof(int),
                              cudaMemcpyDeviceToHost, stream[0]));

    CUDA_CALL(cudaMemcpyAsync(a_d[1], a_h[1], (N / 2) * sizeof(int),
                              cudaMemcpyHostToDevice, stream[1]));
    vectorAddKernel<<<blocks, threads, 0, stream[1]>>>(a_d[1], c_d[1], N);
    CUDA_CALL(cudaMemcpyAsync(c_h[1], c_d[1], (N / 2) * sizeof(int),
                              cudaMemcpyDeviceToHost, stream[1]));

    /* End timers */
    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

    printf("Time to talculate result: %f ms\n", elapsedTime);

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    /* Check the outputs are correct */
    bool failed = false;
    for (int sidx = 0; sidx < NSTREAMS; sidx++) {
        for (int i = 0; i < N; i++) {
            int got = c_h[sidx][i];
            int expected = (a_h[sidx][i] + 1);

            if (got != expected) {
                printf("Error in stream %i element %i: %i != %i\n",
                        sidx, i, got, expected);
                failed = true;
                break;
            }
        }

        if (failed) break;
    }

    for (int i = 0; i < NSTREAMS; i++) {
        CUDA_CALL(cudaStreamDestroy(stream[i]));
        CUDA_CALL(cudaFreeHost(a_h[i]));
        CUDA_CALL(cudaFreeHost(c_h[i]));
    }
    CUDA_CALL(cudaDeviceReset());

    if (failed) {
        exit(1);
    }

    return 0;
}
