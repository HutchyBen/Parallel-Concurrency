#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
__global__ void multiplyNumbers(int* c, const int* a, const int* b) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] * b[index];
}

int main() {
	constexpr int arraySize = 50;
	int* a;
	int* b;
	int* result;

	cudaMallocManaged(&a, sizeof(int) * arraySize);
	cudaMallocManaged(&b, sizeof(int) * arraySize);
	cudaMallocManaged(&result, sizeof(int) * arraySize);

	for (int i = 0; i < arraySize; i++) {
		a[i] = i;
		b[i] = arraySize - i;
	}

	multiplyNumbers<<<5, 10 >>>(result, a, b);
	cudaDeviceSynchronize();

	int dotProduct = 0;
	for (int i = 0; i < arraySize; i++) {
		dotProduct += result[i];
	}

	printf("%d\n", dotProduct);

	cudaFree(a);
	cudaFree(b);
	cudaFree(result);

	return 0;
}