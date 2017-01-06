#include <stdio.h>
#include <cuda.h>
#include <omp.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("\nNumber of GPU devices: %i\n", deviceCount);

    int driverVersion;
    int runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    cudaDeviceProp deviceProperties;
    printf("CUDA driver version / runtime version: %d.%d / %d.%d\n",
            driverVersion / 1000, (driverVersion % 100) / 10,
            runtimeVersion / 1000, (runtimeVersion % 100) / 10);


    for (int i=0; i<deviceCount; i++) {
        cudaGetDeviceProperties(&deviceProperties, i);
        printf("Name: %s\n", deviceProperties.name);
    }


    return 0;
}
