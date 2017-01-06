#include <stdio.h>
#include <cuda.h>
#include <omp.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

__global__ void reduceKernel(int *input, int *output, int N) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    extern __shared__ int sdata[];
    sdata[tid] = 0.0f;

    if (i < N) {
        sdata[tid] = input[i];
        __syncthreads();

        for (int s=1; s<blockDim.x; s*=2) {
            if (tid % (2 * s) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[blockIdx.x] = sdata[0];
        }
    }
}

int nextPow2(int x);

int main() {
    int N = 50000000;

    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);
    int numThreads = deviceProperties.maxThreadsPerBlock;

    printf("Max threads per block: %d\n", numThreads);

    dim3 grid((N + numThreads - 1) / numThreads, 1, 1);
    dim3 block(numThreads, 1, 1);

    printf("Single-GPU reduction sum\n");
    printf("Total number of elements to sum: %i\n", N);
    printf("Kernel launch configuration: %i blocks of %i threads\n", grid.x, block.x);

    int *data_h;
    data_h = (int*)malloc(N * sizeof(int));

    srand(time(NULL));
    for (int i=0; i<N; i++) {
        data_h[i] = rand() % 10;
    }

    int sumCPU = 0;
    printf("Calculating sum on host...\n");
    float startCPU = omp_get_wtime();
    for (int i=0; i<N; i++) sumCPU += data_h[i];
    float endCPU = omp_get_wtime();
    float timeCPU = (endCPU - startCPU) * 1000;

    int *data_d;
    int *blockSum_d;

    cudaMalloc((void**)&data_d, N * sizeof(int));
    cudaMalloc((void**)&blockSum_d, grid.x * sizeof(int));

    cudaMemcpy(data_d, data_h, N * sizeof(int), cudaMemcpyHostToDevice);

    float timeGPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int remainingElements = grid.x;
    int level = 1;

    printf("Level 0 kernel summing %i elements with %i blocks of %i threads...\n",
            N, grid.x, block.x);
    reduceKernel<<<grid, block, block.x * sizeof(int)>>>(data_d, blockSum_d, N);

    while (remainingElements > 1) {
        int numThreads = (remainingElements < block.x) ? nextPow2(remainingElements) : block.x;
        int numBlocks = (remainingElements + numThreads - 1) / numThreads;
        printf("Level %i kernel summing %i elements with %i blocks of %i threads\n",
                level, remainingElements, numBlocks, numThreads);
        reduceKernel<<<numBlocks, numThreads, numThreads * sizeof(int)>>>(blockSum_d, blockSum_d, remainingElements);
        remainingElements = numBlocks;
        level++;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeGPU, start, stop);

    int sumGPU;
    cudaMemcpy(&sumGPU, blockSum_d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("CPU result: %i, processing time: %f ms\n", sumCPU, timeCPU);
    printf("GPU result: %i, processing time: %f ms\n", sumGPU, timeGPU);

    return 0;
}

int nextPow2(int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}
