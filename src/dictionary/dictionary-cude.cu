#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <cuda_runtime.h>
#include "dictionary-util.c"
#include "../hash/hash.h"
#include "../globals.h"

#define TILE_SIZE 256

// CUDA kernel function
__global__ void compare_candidates_kernel(const char *password_hash, const char *dictionary, int *results, int num_candidates, int verbose)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory buffer to store candidate passwords
    __shared__ char shared_candidates[TILE_SIZE][MAX_PASSWORD_LENGTH];

    // Load candidates into shared memory
    if (global_idx < num_candidates)
    {
        strncpy(shared_candidates[threadIdx.x], &dictionary[global_idx * MAX_PASSWORD_LENGTH], MAX_PASSWORD_LENGTH);
    }
    __syncthreads();

    // Perform the comparisons
    if (global_idx < num_candidates)
    {
        int result = do_comparison(password_hash, shared_candidates[threadIdx.x], verbose);
        results[global_idx] = result;
    }
}

int dictionary_crack(char *password_hash, char *dictionary_path, int verbose)
{
    // Read the dictionary file into a buffer
    FILE *file = fopen(dictionary_path, "r");
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *dictionary = (char *)malloc(file_size);
    fread(dictionary, file_size, 1, file);
    fclose(file);

    int num_candidates = file_size / MAX_PASSWORD_LENGTH;

    // Allocate memory on the GPU
    char *dev_password_hash, *dev_dictionary;
    int *dev_results;
    cudaMalloc((void **)&dev_password_hash, sizeof(char) * HASH_LENGTH);
    cudaMalloc((void **)&dev_dictionary, sizeof(char) * file_size);
    cudaMalloc((void **)&dev_results, sizeof(int) * num_candidates);

    // Copy data to the GPU
    cudaMemcpy(dev_password_hash, password_hash, sizeof(char) * HASH_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dictionary, dictionary, sizeof(char) * file_size, cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockDim(TILE_SIZE);
    dim3 gridDim((num_candidates + blockDim.x - 1) / blockDim.x);
    compare_candidates_kernel<<<gridDim, blockDim>>>(dev_password_hash, dev_dictionary, dev_results, num_candidates, verbose);

    // Copy results back to the host
    int *results = (int *)malloc(sizeof(int) * num_candidates);
    cudaMemcpy(results, dev_results, sizeof(int) * num_candidates, cudaMemcpyDeviceToHost);

    // Check the results
    int found = NOT_FOUND;
    for (int i = 0; i < num_candidates; i++)
    {
        if (results[i] == FOUND)
        {
            found = FOUND;
            break;
        }
    }

    // Cleanup
    free(dictionary);
    free(results);
    cudaFree(dev_password_hash);
    cudaFree(dev_dictionary);
    cudaFree(dev_results);

    return found;
}