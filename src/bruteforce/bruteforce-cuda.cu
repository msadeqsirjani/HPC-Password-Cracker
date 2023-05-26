#include <stdio.h>
#include <string.h>
#include "../hash/hash.h"
#include "../globals.h"

#define BATCH_SIZE 10000

__global__ void batch_hash_kernel(char *inputArray, char *outputArray, int password_max_length, int number_of_characters, int possibilities)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i, val;
    char passwordToTest[password_max_length+1];

    while (idx < possibilities)
    {
        val = idx;
        for (i = 0; i < password_max_length; i++)
        {
            passwordToTest[i] = inputArray[(password_max_length*idx) + i];
        }
        passwordToTest[password_max_length] = '\0';

        sha256_encode(passwordToTest, outputArray + (idx*65));

        idx += blockDim.x * gridDim.x;
    }
}

int bruteforce_crack(char *password_hash, char *characters, int password_max_length, int verbose)
{
    // Input Calculations
    int number_of_characters = strlen(characters);
    if (verbose)  print_stats(password_hash, characters, number_of_characters, password_max_length);

    // Program counters and flags
    int i, j, result;
    result = NOT_FOUND;
    for (i = 1; i <= password_max_length; i++)
    {
        long possibilities = calculate_possibilities(number_of_characters, i, verbose, 0);
        int num_batches = (possibilities + BATCH_SIZE - 1) / BATCH_SIZE;

        char *d_inputArray, *d_outputArray;
        cudaMalloc(&d_inputArray, BATCH_SIZE * i);
        cudaMalloc(&d_outputArray, BATCH_SIZE * 65);

        for (j = 0; j < num_batches; j++)
        {
            int batch_size = (j == num_batches-1) ? (possibilities % BATCH_SIZE) : BATCH_SIZE;
            int num_blocks = (batch_size + 255) / 256;
            int num_threads = (batch_size < 256) ? batch_size : 256;
            int shared_mem_size = i * num_threads;

            char inputArray[BATCH_SIZE * i];
            char outputArray[BATCH_SIZE * 65];

            for (int k = 0; k < batch_size; k++)
            {
                int val = j*BATCH_SIZE + k;
                for (int l = 0; l < i; l++)
                {
                    inputArray[(i*k) + l] = characters[val % number_of_characters];
                    val = val / number_of_characters;
                }
            }

            cudaMemcpy(d_inputArray, inputArray, batch_size * i, cudaMemcpyHostToDevice);

           batch_hash_kernel<<<num_blocks, num_threads, shared_mem_size>>>(d_inputArray, d_outputArray, i, number_of_characters, batch_size);
            cudaDeviceSynchronize();

            cudaMemcpy(outputArray, d_outputArray, batch_size * 65, cudaMemcpyDeviceToHost);

            for (int k = 0; k < batch_size; k++)
            {
                if (!strcmp(password_hash, outputArray + k*65))
                {
                    printf("Password found: %s\n", outputArray + k*65);
                    result = FOUND;
                    cudaFree(d_inputArray);
                    cudaFree(d_outputArray);
                    return result;
                }
            }
        }
        cudaFree(d_inputArray);
        cudaFree(d_outputArray);
    }
    if (result == NOT_FOUND)
    {
      printf("Password not found.\n");
    }
    return result;
}