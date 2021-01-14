#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include <iostream>

#include "Watermarking_CUDA.h"

// Numer of threads

// Cuda error handling
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d -- %s\n",__FILE__,__LINE__, cudaGetErrorString(x)); \
    return;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
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