#include <stdio.h>
#include <cuda.h>
#include <omp.h>
#define N 256
#define TILE_WIDTH 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

__global__ void matrix_add_kernel(int *d_a, int *d_b, int *d_c) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    int index = row * N + col;

    if ((row < N) && (col < N)) {
        int aval = d_a[index];
        int bval = d_b[index];
        int result = aval + bval;
        /* printf("row %i col %i a: %d, b: %d, result: %d\n", row, col, aval, bval, result); */
        d_c[index] = result;
    }
}

__global__ void matrix_mult_kernel(int *d_a, int *d_b, int *d_c) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    /* int index = row * N + col; */


    if ((row < N) && (col < N)) {
        int result = 0;
        for (int k=0; k<N; k++) {
            result += d_a[row * N + k] * d_b[k * N + col];
        }
        d_c[row * N + col] = result;
    }
}

__global__ void matrix_mult_kernel_optimised(int *d_a, int *d_b, int *d_c) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = ty + blockDim.y * blockIdx.y;
    int col = tx + blockDim.x * blockIdx.x;

    if ((row < N) && (col < N)) {

        __shared__ int s_a[TILE_WIDTH][TILE_WIDTH];
        __shared__ int s_b[TILE_WIDTH][TILE_WIDTH];
        int result = 0;

        for (int p=0; p<N/TILE_WIDTH; p++) {
            s_a[ty][tx] = d_a[row * N + (p * TILE_WIDTH + tx)];
            s_b[ty][tx] = d_b[(p * TILE_WIDTH + ty) * N + col];

            __syncthreads();

            for (int i=0; i<TILE_WIDTH; i++) {
                result += s_a[ty][i] * s_b[i][tx];
            }

            __syncthreads();
        }

        d_c[row * N + col] = result;
    }
}

void matrix_mult(int *a, int *b, int *c) {
    int row, col, k, sum;
    for (row=0; row<N; row++) {
        for (col=0; col<N; col++) {
            sum = 0;
            for (k=0; k<N; k++) {
                sum += a[row * N + k] * b[k * N + col];
            }
            c[row * N + col] = sum;
        }
    }
}

void matrix_add(int *a, int *b, int *c) {
    int index;
    for (int col=0; col<N; col++) {
        for (int row=0; row<N; row++) {
            index = row * N + col;
            c[index] = a[index] + b[index];
        }
    }
}

void print_matrix(int *matrix, const char *name) {
    printf("Matrix %s\n", name);
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            printf("(%i,%i) = %i\n", j, i, matrix[i * N + j]);
        }
    }
}

int main() {
    int *h_a, *h_b, *h_c, *h_d;
    int *d_a, *d_b, *d_c;

    int size = N * N * sizeof(int);
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);
    h_d = (int*)malloc(size);

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            h_a[i * N + j] = i + 1;
            h_b[i * N + j] = i + 1;
            h_c[i * N + j] = 0;
        }
    }

    /* print_matrix(h_a, "Host input A"); */
    /* print_matrix(h_b, "Host input B"); */

    gpuErrchk(cudaMalloc(&d_a, size));
    gpuErrchk(cudaMalloc(&d_b, size));
    gpuErrchk(cudaMalloc(&d_c, size));

    gpuErrchk(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    float deviceElapsedTime, hostElapsedtime;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, 0));

    dim3 grid(N, N);
    dim3 block(TILE_WIDTH, TILE_WIDTH);

    matrix_mult_kernel_optimised<<<grid, block>>>(d_a, d_b, d_c);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaEventRecord(stop, 0));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&deviceElapsedTime, start, stop));

    gpuErrchk(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    printf("Time taken: %f ms\n", deviceElapsedTime);

    gpuErrchk(cudaEventRecord(start, 0));

    matrix_mult(h_a, h_b, h_d);

    /* print_matrix(h_d, "Host result"); */
    /* print_matrix(h_c, "Device result"); */

    gpuErrchk(cudaEventRecord(stop, 0));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&hostElapsedtime, start, stop));

    printf("Time taken: %f ms\n", hostElapsedtime);

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    for (int i=0; i<N*N; i++) {
        if (h_c[i] != h_d[i]) {
            printf("Error: CPU and GPU results do not match at index %d, %d != %d\n",
                    i, h_c[i], h_d[i]);
            exit(1);
        }
    }

    printf("Speedup: %f\n", hostElapsedtime / deviceElapsedTime);

    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_b));
    gpuErrchk(cudaFree(d_c));

    gpuErrchk(cudaDeviceReset());
    return 0;
}
