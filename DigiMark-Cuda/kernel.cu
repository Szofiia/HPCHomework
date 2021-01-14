#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#include <iostream>

#include "Watermarking_CUDA.h"

// Cuda error handling
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d -- %s\n",__FILE__,__LINE__, cudaGetErrorString(x)); \
    return;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return;}} while(0)

#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return;}} while(0)

// Calculates curandState.
//
// state - state to be generated
// seed - unified state
// N - number of random states generated
__global__ void setup_kernel_for_random(curandState* state, unsigned long seed, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    curand_init(seed, idx, 0, &state[idx]);
}

// Calculates random values in the device
//
// global_state - global state used for generation
// random values - array containing the random values
__global__ void generate_for_random(curandState* global_state, float* random_values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = global_state[idx];
    float random = curand_uniform(&localState);
    random_values[idx] = random;
    global_state[idx] = localState;
}

// Adds a little to each pixel
__global__ void add_watermark(float* data, int N)
{
    //extern __shared__ int sdata[];
    
    int tidx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //sdata[tidx] = data[idx];
    //__syncthreads();

   /* for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (tidx < i)
        {

            data[tidx] += 0.5f;
        }
        __syncthreads();
    }*/

    if (idx < N)
        data[idx] += 0.5f;
}

// Calculates random values on the device and copies to the host,
// using curand Host API. (curandGenerator)
//
// host_data - copies random values to this container
// N - number of random values generated
void CalcRandWithHostAPI(float* host_data, int N)
{
    float *dev_data;
    curandGenerator_t generator;

    CUDA_CALL(cudaMalloc((void**)&dev_data, N * sizeof(float)));

    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL));
    CURAND_CALL(curandGenerateUniform(generator, dev_data, N));

    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(host_data, dev_data, N * sizeof(*host_data), cudaMemcpyDeviceToHost));

    // Cleanup
    CURAND_CALL(curandDestroyGenerator(generator));
    CUDA_CALL(cudaFree(dev_data)); 
}

// Calculates random values on the device and copies to the host,
// using curand Device API. (curandState)
//
// host_data - copies random values to this container
// N - number of random values generated
void CalcRandWithDevAPI(float* host_data, int N)
{
    // 2D Thread blocks
    dim3 threads;
    if (N > 1024)
    {
        threads = dim3(512, 1);
    }
    else
    {
        threads = dim3(N, 1);
    }

    int blocks_count = floor(N / threads.x);
    dim3 blocks = dim3(blocks_count, 1);

    curandState* dev_states;
    float* dev_random_values;

    CUDA_CALL(cudaMalloc(&dev_states, N * sizeof(curandState)));
    CUDA_CALL(cudaMalloc(&dev_random_values, N * sizeof(*host_data)));

    // Setting up the random state
    setup_kernel_for_random << <blocks, threads >> > (dev_states, time(NULL), N);

    // Generating random numbers
    generate_for_random << <blocks, threads >> > (dev_states, dev_random_values);

    CUDA_CALL(cudaMemcpy(host_data, dev_random_values, N * sizeof(*host_data), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CALL(cudaFree(dev_states));
    CUDA_CALL(cudaFree(dev_random_values));
}

// Calculates the watermark in the mos significant blocks
//
// host_data - the input container with values
// N - size of the input container
// w - the watermark
// alpha - allpha for embedding the watermark
void CalcWatermark(float* host_data, int N, float w, float alpha) 
{
    float *dev_data, *temp_data;
    int max_index = 0;
    cublasHandle_t my_handle;
    CUBLAS_CALL(cublasCreate(&my_handle));
    
    // Allocate temporary for max
    temp_data = new float[N * sizeof(float)];
    memcpy(temp_data, host_data, N * sizeof(float));

    CUDA_CALL(cudaMalloc((void**)&dev_data, N * sizeof(float)));

    CUDA_CALL(cudaMemcpy(dev_data, host_data, N * sizeof(*host_data), cudaMemcpyHostToDevice));
    CUBLAS_CALL(cublasIsamax(my_handle, N, dev_data, 1, &max_index));
    temp_data[max_index] = 0;
    //CUDA_CALL(cudaFree(dev_data));

    CUDA_CALL(cudaMemcpy(dev_data, temp_data, N * sizeof(*host_data), cudaMemcpyHostToDevice));
    CUBLAS_CALL(cublasIsamax(my_handle, N, dev_data, 1, &max_index));
    host_data[max_index] = host_data[max_index] + alpha * w;
    CUDA_CALL(cudaFree(dev_data));
}