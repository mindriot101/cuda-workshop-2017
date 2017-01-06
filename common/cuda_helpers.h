#ifndef CUDA_HELPERS_H_
#define CUDA_HELPERS_H_

#include <assert.h>

#define CUDA_CALL(x) { \
    const cudaError_t a = (x); \
    if (a != cudaSuccess) { \
        printf("\n CUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(a), a); \
        cudaDeviceReset(); \
        assert(0); \
    } \
}

inline void cuda_peek() {
    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("\n%s\n", cudaGetErrorString(cudaGetLastError()));
        cudaDeviceReset();
    }
}

#endif //  CUDA_HELPERS_H_
