## Exercise 1: Thread Index Identification in CUDA 

### Question

For the vector addition problem considered in the CUDA template, list the values for the built-in variables `threadIdx.x` (the thread's index within its block) and `blockIdx.x` (the block's index within the grid) corresponding to the following thread configurations used for executing the kernel `addKernel()` function on the GPU :

1. `addKernel<<<1, 5>>>(dev_c, dev_a, dev_b);` 
2. `addKernel<<<2, 3>>>(dev_c, dev_a, dev_b);` 
3. `addKernel<<<3, 2>>>(dev_c, dev_a, dev_b);` 
4. `addKernel<<<6, 1>>>(dev_c, dev_a, dev_b);` 



### Solution
Updated the default addKernel to print thread index and block index
```c
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    int block = blockIdx.x;
    printf("id: %d   -   block: %d\n", i, block);
    c[i] = a[i] + b[i];
}
```

### Sample Output
#### `addKernel<<<1, 5>>>(dev_c, dev_a, dev_b);` 
```
id: 0   -   block: 0
id: 1   -   block: 0
id: 2   -   block: 0
id: 3   -   block: 0
id: 4   -   block: 0
```

#### `addKernel<<<2, 3>>>(dev_c, dev_a, dev_b);` 
```
id: 0   -   block: 1
id: 1   -   block: 1
id: 2   -   block: 1
id: 0   -   block: 0
id: 1   -   block: 0
id: 2   -   block: 0
```
#### `addKernel<<<3, 2>>>(dev_c, dev_a, dev_b);` 
```
id: 0   -   block: 1
id: 1   -   block: 1
id: 0   -   block: 2
id: 1   -   block: 2
id: 0   -   block: 0
id: 1   -   block: 0
```
#### `addKernel<<<6, 1>>>(dev_c, dev_a, dev_b);` 
```
id: 0   -   block: 1
id: 0   -   block: 2
id: 0   -   block: 5
id: 0   -   block: 0
id: 0   -   block: 4
id: 0   -   block: 3
```
### Reflection


## Exercise 2: Global Thread Index for Multiple 1D Thread Blocks 

### Question

In the vector addition example provided in the CUDA programming template, determine the resulting output when the kernel is executed with the following thread configuration:

`addKernel<<<3, 2>>>(dev_c, dev_a, dev_b);` 

To accomplish this, modify the following line within the CUDA kernel so that it correctly reflects the global thread index rather than just the local index:

```cpp
int i = threadIdx.x; // Modify this line

```



### Solution
Set the thread configuration to 3 blocks, 2 threads per block.
```c
addKernel<<<3, 2>>>(dev_c, dev_a, dev_b);
```
```c
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

```
Expanded the test data to 6 items
```c
const int arraySize = 6;
const int a[arraySize] = { 1, 2, 3, 4, 5, 6 };
const int b[arraySize] = { 10, 20, 30, 40, 50, 60 };

```
### Sample Output
`{1,2,3,4,5,6} + {10,20,30,40,50,60} = {11,22,33,44,55,66}`
### Reflection

---

## Exercise 3: 2D Thread Blocks 

### Question

CUDA supports 1D, 2D, and 3D block and grid configurations using the `dim3` type . For each of the following thread configurations used to launch the `addKernel()` function on the GPU, list all possible values of the built-in variables `threadIdx.x` and `threadIdx.y` :

1. `addKernel<<<1, dim3(2, 3)>>>(dev_c, dev_a, dev_b);` 


2. `addKernel<<<1, dim3(3, 3)>>>(dev_c, dev_a, dev_b);` 



### Solution
```c
__global__ void addKernel(int *c, const int *a, const int *b, const int size)
{
    printf("x: %d  -  y: %d\n", threadIdx.x, threadIdx.y);
    int i = threadIdx.x + threadIdx.y*size;
    c[i] = a[i] + b[i];
}
```


### Sample Output
#### `addKernel<<<1, dim3(2, 3)>>>(dev_c, dev_a, dev_b);`
```
x: 0  -  y: 0
x: 1  -  y: 0
x: 0  -  y: 1
x: 1  -  y: 1
x: 0  -  y: 2
x: 1  -  y: 2
```
#### `addKernel<<<1, dim3(3, 3)>>>(dev_c, dev_a, dev_b);` 
```
x: 0  -  y: 0
x: 1  -  y: 0
x: 2  -  y: 0
x: 0  -  y: 1
x: 1  -  y: 1
x: 2  -  y: 1
x: 0  -  y: 2
x: 1  -  y: 2
x: 2  -  y: 2
```

### Reflection

---

## Exercise 4: Global Thread Index for 2D Thread Blocks 

### Question

In the vector addition example provided in the CUDA programming template, determine the resulting output when the kernel is executed with the following 2D thread configuration:

`addKernel<<<1, dim3(3, 2)>>>(dev_c, dev_a, dev_b);` 

To accomplish this, modify the following line within the CUDA kernel so that it calculates a unique index for the data elements based on the 2D thread structure:

```cpp
int i = threadIdx.x; // Modify this line

```



### Solution
Updated input to be 2D for readability, passed to addWithCuda flattened however
```
const int arraySize = 3;
const int a[arraySize][arraySize] = { {1,2,3},{4,5,6},{7,8,9} };
const int b[arraySize][arraySize] = { {10,20,30},{40,50,60},{70,80,90} };
int c[arraySize][arraySize] = { {0,0,0} };
cudaError_t cudaStatus = addWithCuda(&c[0][0], &a[0][0], &b[0][0], arraySize);
```

Updated malloc's and memcpy to adjust for 3x3
```c
cudaStatus = cudaMalloc((void**)&dev_a, size*size * sizeof(int));
...
cudaStatus = cudaMemcpy(dev_a, a, size * size * sizeof(int), cudaMemcpyHostToDevice);
```

Added a size parameter to the addKernel so dimensions are known
```
__global__ void addKernel(int *c, const int *a, const int *b, const int size)
{
    printf("x: %d  -  y: %d\n", threadIdx.x, threadIdx.y);
    int i = threadIdx.x + threadIdx.y*size;
    c[i] = a[i] + b[i];
}
```

### Sample Output
```c
x: 0  -  y: 0
x: 1  -  y: 0
x: 2  -  y: 0
x: 0  -  y: 1
x: 1  -  y: 1
x: 2  -  y: 1
11 22 33
44 55 66
0 0 0
```
### Reflection

---

## Exercise 5: Global Thread Index for Multiple 2D Thread Blocks 

### Question

In the vector addition example provided in the CUDA programming template, determine the resulting output when the kernel is executed with a grid of 2D blocks using the following configuration:

`addKernel<<<dim3(2, 3), dim3(2, 2)>>>(dev_c, dev_a, dev_b);` 

To accomplish this, modify the following line within the CUDA kernel so that it correctly reflects the global thread index across multiple 2D blocks:

```cpp
int i = threadIdx.x; // Modify this line

```



### Solution
```

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = blockIdx.x * 12 + blockIdx.y*4 + threadIdx.x*2 + threadIdx.y;
    c[i] = a[i] + b[i];
}

int main()
{
    /*const int arraySize = 6;
    const int a[arraySize] = { 1, 2, 3, 4, 5, 6 };
    const int b[arraySize] = { 10, 20, 30, 40, 50, 60 };
    int c[arraySize] = { 0 };*/


    /*const int arraySize = 3;
    const int a[arraySize][arraySize] = { {1,2,3},{4,5,6},{7,8,9} };
    const int b[arraySize][arraySize] = { {10,20,30},{40,50,60},{70,80,90} };
    int c[arraySize][arraySize] = { {0,0,0} };*/


    // GEMINI USED TO GENERATE ARRAYS BECAUSE I DONT HATEMYSELF
    int a[2][3][2][2] = {
        { // Outer Row 0
            {{1, 2}, {3, 4}}, // 2x2 Grid at [0][0]
            {{5, 6}, {7, 8}}, // 2x2 Grid at [0][1]
            {{9, 0}, {1, 2}}  // 2x2 Grid at [0][2]
        },
        { // Outer Row 1
            {{3, 4}, {5,6}}, // 2x2 Grid at [1][0]
            {{7, 8}, {9, 0}}, // 2x2 Grid at [1][1]
            {{1, 2}, {3, 4}}  // 2x2 Grid at [1][2]
        }
    };




    int b[2][3][2][2] = {
    { // Outer Row 0
        {{10, 20}, {30, 40}}, // [0][0]
        {{50, 60}, {70, 80}}, // [0][1]
        {{90,  0}, {10, 20}}  // [0][2]
    },
    { // Outer Row 1
        {{30, 40}, {50, 60}}, // [1][0]
        {{70, 80}, {90,  0}}, // [1][1]
        {{10, 20}, {30, 40}}  // [1][2]
    }
    };

    int c[2][3][2][2] = {};

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(&c[0][0][0][0], &a[0][0][0][0], &b[0][0][0][0]);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // PRINTF GENERATED BY GEMINI BECASE I DONT HATE YMSELF
    printf(
        "--- ROW 0 ---\n"
        "[%2d %2d]  [%2d %2d]  [%2d %2d]\n"
        "[%2d %2d]  [%2d %2d]  [%2d %2d]\n\n"
        "--- ROW 1 ---\n"
        "[%2d %2d]  [%2d %2d]  [%2d %2d]\n"
        "[%2d %2d]  [%2d %2d]  [%2d %2d]\n",
        // Row 0, Box 0       Row 0, Box 1       Row 0, Box 2
        c[0][0][0][0], c[0][0][0][1], c[0][1][0][0], c[0][1][0][1], c[0][2][0][0], c[0][2][0][1],
        c[0][0][1][0], c[0][0][1][1], c[0][1][1][0], c[0][1][1][1], c[0][2][1][0], c[0][2][1][1],

        // Row 1, Box 0       Row 1, Box 1       Row 1, Box 2
        c[1][0][0][0], c[1][0][0][1], c[1][1][0][0], c[1][1][0][1], c[1][2][0][0], c[1][2][0][1],
        c[1][0][1][0], c[1][0][1][1], c[1][1][1][0], c[1][1][1][1], c[1][2][1][0], c[1][2][1][1]
    );

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b)
{

    int *dev_a;
    int *dev_b;
    int *dev_c;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, 2 * 2 * 2 * 3 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, 2 * 2 * 2 * 3 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, 2 * 2 * 2 * 3 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, 2 * 2 * 2 * 3 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "A cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, 2 * 2 * 2 * 3 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "B cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<dim3(2,3), dim3(2, 2)>>>(dev_c, dev_a, dev_b);

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
    cudaStatus = cudaMemcpy(c, dev_c, 2 * 2 * 2 * 3 * sizeof(int), cudaMemcpyDeviceToHost);
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
```
### Test Data

### Sample Output

### Reflection

---

## Exercise 6: Matrix Addition (Optional) 

### Question

Write a CUDA program that computes the matrix sum  by performing the addition in parallel on the GPU . Consider two  matrices  and , defined procedurally by their entries at row  and column  (using zero-based indexing) as follows:


Your program must complete the following steps:

1. Allocate memory for , , and  on both the host and device.


2. Initialize  and  according to the definitions above.


3. Launch a CUDA kernel that computes  for all .


4. Copy the result matrix  back to the host and verify correctness (e.g., by checking selected entries or printing a small portion of the output).



### Solution

### Test Data

### Sample Output

### Reflection