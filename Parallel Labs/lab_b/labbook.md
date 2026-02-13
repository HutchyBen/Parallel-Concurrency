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
In the first line which creates 5 threads per block with one thread, it creates a total of 5 threads, where the block will always be 0 and thread index will go from 0 - 5. For the rest it has a total of 6 threads, where number 2 has 2 blocks each with 3 threads, third has 3 blocks with two threads in each one and the final line has 6 blocks each with one thread.

For this example where it's a 1D block and 1D grid of blocks, only the x-axis is used for threadIdx, and blockIdx

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
To run my code with the configuration of 3 blocks with 2 threads per block the index used for accessing the array needs to be changed, what it needs to be changed to is the `x block index * x block dimension + x thread index`.

This leads to it counting to through all the indexes ensuring that the blocks are considered and all 6 indexes are counted  

I don't yet have to think of the dimensions as being greater than 1D so I only have to worry about the x-axis at the moment.



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
    int i = threadIdx.x + threadIdx.y*blockDim.x;
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
For working with a 2D block, you pass a dim3 to the configuration to tell CUDA you want a multidimensional block. You use the threadIdx.x and threadIdx.y (then threadIdx.z if working in 3d) to access the index of thread in the block. You also use blockDim to get the dimensions of the block.

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
__global__ void addKernel(int *c, const int *a, const int *b)
{
    printf("x: %d  -  y: %d\n", threadIdx.x, threadIdx.y);
    int i = threadIdx.x + threadIdx.y*blockDim.x;
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
0 0 0 // this is fine my grid was 3x3 for other data this example is calculating 3x2
```
### Reflection
To get the global index for a 2D block, the calculation is `x index + (y index * size of row)`. This looks like `int i = threadIdx.x + (threadIdx.y * blockDim.x)`. This leads to the correct global index being obtained.

I was initially confused about `blockDim` and was passing the size to the kernel however after monday's lecture I understood how `blockDim` and `gridDim` works.



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
__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x + (threadIdx.y * blockDim.x) + (blockDim.x * blockDim.y * blockIdx.x) + (blockIdx.y * blockDim.x * blockDim.y * gridDim.x);
    c[i] = a[i] + b[i];
};
```
### Test Data

```c
// GEMINI USED TO GENERATE THIS
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
```

### Sample Output
```
--- ROW 0 ---
[11 22]  [55 66]  [99  0]
[33 44]  [77 88]  [11 22]

--- ROW 1 ---
[33 44]  [77 88]  [11 22]
[55 66]  [99  0]  [33 44]

```
### Reflection
For getting the global index when theres a 2D block and 2D grid, it quite a decent bit of calculation to do. The way i think about it was, first get the index of the block you want using `threadIdx.x + threadIdx.y * blockDim.x`. Then figure out the index start index of the block inside the 2D grid. First I figured out the x-axis by doing `blockDim.x * blockDim.y * blockIdx.x`, I then need to add the offset to get to the right y-axis. This looks like `blockDim.x * blockDim.y * gridDim.x * blockIdx.y `. Adding the offset inside block, offset for x-axis and offset for y-axis gives the global index for the thread.

This gave me a headache trying to keep track of all the dimensions but I now understand how all the grid, block and variables all work together to represent the position of a thread.