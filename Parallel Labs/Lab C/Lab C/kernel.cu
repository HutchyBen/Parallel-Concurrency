#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
__global__ void multiplyNumbers(int* result, const int* a, const int* b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    result[idx] = a[idx] * b[idx];
}

int main() {
    constexpr int arraySize = 50;
    int vectorA[arraySize] = { 0 };
    int vectorB[arraySize] = {0};
    int cpuResult[arraySize] = { 0 };
    
    for (int i = 0; i < arraySize; i++) {
        vectorA[i] = i;
        vectorB[i] = arraySize - i;
    }

    int* a;
    int* b;
    int* result;
    cudaMalloc(&a, sizeof(int) * arraySize);
    cudaMalloc(&b, sizeof(int) * arraySize);
    cudaMalloc(&result, sizeof(int) * arraySize);

    cudaMemcpy(a, vectorA, sizeof(int) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(b, vectorB, sizeof(int) * arraySize, cudaMemcpyHostToDevice);

    multiplyNumbers << <5, 10 >> > (result, a, b);
    cudaDeviceSynchronize();
    cudaMemcpy(cpuResult, result, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);

    int dotProduct = 0;

    for (int i = 0; i < arraySize; i++) {
        dotProduct += cpuResult[i];
    }

    printf("%d\n", dotProduct);

    cudaFree(a);
    cudaFree(b);
    cudaFree(result);
	return 0;
}