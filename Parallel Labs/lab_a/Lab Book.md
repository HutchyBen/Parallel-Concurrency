# 500083 Lab Book

## Week 1 - Lab A CUDA

28 Jan 2026

### Q1. Set up CUDA Project in Visual Studio 2022

**Question:**

Create a solution in visual studio 2022

**Reflection:**
Visual Studio has a built in template for CUDA projects, this project type provides example code that will perform basic addition of two vectors on the GPU

Another part of the template ensures that the correct NVIDIA compiler is used when building

### Q2. Understanding the CUDA Programming Model

**Reflection:**
When using CUDA the workflow consists of, preparing data on the CPU, allocating and copying the data to the GPU, telling the GPU to perform calculations then copying the results back to the CPU

When defining a function you use the __global__ keyword to say to the compiler that the code will be ran on the GPU. This provides the threadId variable within the function which gives you access to read which thread number the function is which can be used for array indexes when procesing arrays.

Managing memory on the GPU requires using CUDA provided functions, the common ones used are `cudaMalloc` to allocate memory on the GPU and get the address for a pointer, this takes a pointer to a pointer and the size of memory to allocate. The next function used is `cudaMemcpy` which is used for copying memory between devices. It takes a destination, source, size and a copy mode. The copy mode consists of different enum options to state if its Host (CPU) to Device (GPU), or any combination of these.

To execute code on the GPU, you use the execution configuration syntax ```<<<blockCount, threadsPerBlock>>>```. This syntax provides the information to CUDA to run the function in parallel on the GPU.

### Q3. Refactoring for Heterogeneous Understanding
**Question**
Rewrite the existing addWithCuda function to execute in the main function to help understand the workflow.

**Solution**
```c
int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t error;

    error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        fprintf(stderr, "failed to set device");
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
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "error occured with addKernel");
        goto CleanUp;
    }


    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "failed to synchronize device");
        goto CleanUp;
    }

    error = cudaMemcpy(c, dev_c, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);


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
```

**Output**
```
{1,2,3,4,5} + {10,20,30,40,50} = {11,22,33,44,55}
```

**Reflection**
The code to run code on the GPU consists of 
- Defining arrays on the CPU, these are used as input and output
- Set the CUDA device with `cudaSetDevice`. This tells CUDA which GPU to use for processing
- Allocate memory on the GPU with `cudaMalloc` and store the addresses in seperate pointers
- Copy the arrays from the CPU onto the GPU with `cudaMemcpy`
- Execute the code on the GPU specifying thread count to match the size of arrays
- Wait for all threads to execute with `cudaDeviceSynchronize`
- Copy result back to the CPU with `cudaMemcpy`
- Free all memory with `cudaFree`

Overall the code is not yet too difficult to understand and all memory management is nearly the same as C memory management with a minor difference in `malloc` is you pass the pointer to the address instead of it returning the address. 



### Error handling and synchronization
**Question**
How does error handling work? and why is synchronization needed?
**Reflection**
The error handling in CUDA works by CUDA functions returning a status which can be compared with cudaSuccess to determine if an error has occured.
This is needed as if errors and manually checked, execution will continue when the GPU could be in a state where it can't continue execution such as some memory not being allocated causing issues when copying data to the GPU.

Synchronization occurs with cudaDeviceSynchronize which will wait for all threads to finish before continuing execution. This is needed as if we continue without all threads being finished there may be data that hasn't been processed yet.

### Performance Profiling
**Question**
Implement profiling in the code to determine the effect of increasing the arraySize

**Solution**
```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
addKernel<<<1, arraySize >>>(dev_c, dev_a, dev_b);
error = cudaGetLastError();
if (error != cudaSuccess) {
    fprintf(stderr, "error occured with addKernel");
    goto CleanUp;
}
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Time taken: %f ms\n", milliseconds);
error = cudaDeviceSynchronize();
if (error != cudaSuccess) {
    fprintf(stderr, "failed copy dev_c");
    goto CleanUp;
}
```
**Output**
*10 items*
Time taken: 0.612096 ms

*1000 items*
Time taken: 0.608384 ms

**Reflection**
This shows that the time to execute is noticably not different when increasing the arraySize, this is because the GPU is capable of executing large amounts of data in parallel. The difference in execution is likely due to overhead in starting the kernel.
