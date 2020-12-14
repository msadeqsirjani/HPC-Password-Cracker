#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "bruteforce-util.c"
#include "../hash/hash.h"
#include "../globals.h"

#define BATCH_SIZE 10000

int bruteforce_crack(char *password_hash, char *characters, int password_max_length, int verbose)
{
    // Input Calculations
    int number_of_characters = strlen(characters);
    if (verbose)  print_stats(password_hash, characters, number_of_characters, password_max_length);

    // Program counters and flags
    int i, j, k, l, result;
    result = NOT_FOUND;
    for (i = 1; i <= password_max_length; i++)
    {
        long possibilities = calculate_possibilities(number_of_characters, i, verbose, 0);
        char passwordToTest[i+1];
        for (j = 0; j < possibilities;)
        {
            int nextStep = j + BATCH_SIZE;
            char inputArray[BATCH_SIZE * i];
            char outputArray[BATCH_SIZE * 65];
            for(k = 0; k < BATCH_SIZE && k+j < possibilities; k++) 
            {
                int val = j + k;
                for (l = 0; l < i; l++)
                {
                    inputArray[(i*k) + l] = characters[val % number_of_characters];
                    val = val / number_of_characters;
                }
            }

            //printf("%.*s\n", BATCH_SIZE * i, inputArray);
            batch_hash(inputArray, outputArray, i, BATCH_SIZE);
            
            for(k = 0; k < BATCH_SIZE && k+j < possibilities; k++) 
            {
                //printf("Password: %s\n", outputArray + k*65);
                if (!strcmp(password_hash, outputArray + k*65))
                {
                    printf("Password found: %s\n", outputArray + k*65);
                    result = FOUND;
                    return result; // want to break out of these nested for loops...
                }
            }
            j+=k;
        }
    }
    if (result == NOT_FOUND)
    {
      printf("Password not found.\n");
    }
    return result;
}