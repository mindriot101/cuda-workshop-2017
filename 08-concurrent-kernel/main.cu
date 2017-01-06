#include <stdio.h>
#include <cuda.h>
#include <omp.h>
#include "cuda_helpers.h"

#define NUM_KERNELS 8
#define NUM_STREAMS 9

__global__ void clockBlockKernel(clock_t *output_d, clock_t clockCount) {
    clock_t startClock = clock();
    clock_t clockOffset = 0;

    while (clockOffset < clockCount) {
        clockOffset = clock() - startClock;
    }

    output_d[0] = clockOffset;
}

__global__ void sumKernel(clock_t *clocks_d, int N) {
    __shared__ clock_t clocks_s[32];
    clock_t sum = 0;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += clocks_d[i];
    }

    clocks_s[threadIdx.x] = sum;
    syncthreads();

    for (int i = 16; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            clocks_s[threadIdx.x] += clocks_s[threadIdx.x + 1];
        }
        syncthreads();
    }

    clocks_d[0] = clocks_s[0];
}

int main() {
    float kernelTime = 10;
    float elapsedTime;
    int device = 0;

    cudaDeviceProp deviceProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);

    if (deviceProp.concurrentKernels == 0) {
        printf(" - GPU does not support concurrent kernel execution\n");
        exit(EXIT_FAILURE);
    }

    int clockRate = deviceProp.clockRate;
    printf(" - device clock rate: %.3f Ghz\n", (float)clockRate / 1000000);

    clock_t *a_h = 0;
    cudaMallocHost((void **)&a_h, NUM_KERNELS * sizeof(clock_t));

    clock_t *a_d = 0;
    cudaMalloc((void **)&a_d, NUM_KERNELS * sizeof(clock_t));

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&(streams[i]));
    }

    cudaEvent_t kernelEvent[NUM_KERNELS];
    for (int i = 0; i < NUM_KERNELS; i++) {
        cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming);
    }

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    clock_t totalClocks = 0;
    clock_t timeClocks = (clock_t)(kernelTime * clockRate);

    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < NUM_KERNELS; i++) {
        clockBlockKernel<<<1, 1, 0, streams[i]>>>(&a_d[i], timeClocks);
        cuda_peek();
        totalClocks += timeClocks;
        cudaEventRecord(kernelEvent[i], streams[i]);
        cudaStreamWaitEvent(streams[NUM_STREAMS - 1], kernelEvent[i], 0);
    }

    sumKernel<<<1, 32, 0, streams[NUM_STREAMS - 1]>>>(a_d, NUM_KERNELS);
    cudaMemcpyAsync(a_h, a_d, sizeof(clock_t), cudaMemcpyDeviceToHost,
                    streams[NUM_STREAMS - 1]);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    printf("Expected time for serial execution of %d kernels = %.3fms\n",
           NUM_KERNELS, NUM_KERNELS * kernelTime);
    printf("Expected time for concurrent execution of %d kernels = %.3fms\n",
           NUM_KERNELS, kernelTime);
    printf("Measured time for sample = %.3fms\n", elapsedTime);

    bool correctResult = (a_h[0] > totalClocks);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFreeHost(a_h);
    cudaFree(a_d);

    for (int i=0; i<NUM_KERNELS; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(kernelEvent[i]);
    }

    cudaDeviceReset();

    if (!correctResult) {
        printf("FAILED\n");
        exit(EXIT_FAILURE);
    }

    printf("PASSED\n");

    return 0;
}
