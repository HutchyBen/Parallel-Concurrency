
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };






    // Add vectors in parallel.
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t error;

    error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto CleanUp;
    }

    error = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    if (error != cudaSuccess) {
        fprintf(stderr, "failed alloc dev_a");
        goto CleanUp;
    }
    error = cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    if (error != cudaSuccess) {
        fprintf(stderr, "failed alloc dev_b");
        goto CleanUp;
    }
    error = cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
    if (error != cudaSuccess) {
        fprintf(stderr, "failed alloc dev_c");
        goto CleanUp;
    }
    

    error = cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "failed copy dev_a");
        goto CleanUp;
    }
    error = cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "failed copy dev_b");
        goto CleanUp;
    }
    error = cudaMemcpy(dev_c, c, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "failed copy dev_c");
        goto CleanUp;
    }

    addKernel<<<1, arraySize >>>(dev_c, dev_a, dev_b);
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "failed to synchronize device");
        goto CleanUp;
    }

    error = cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr, "failed to copy result");
        goto CleanUp;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

CleanUp:
    error = cudaFree(dev_a);
    error = cudaFree(dev_b);
    error = cudaFree(dev_c);

    if (error != cudaSuccess) {
        fprintf(stderr, "failed to free memory");
        return 1;
    }

    error = cudaDeviceReset();
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
