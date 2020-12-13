#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../hash/hash.h"
#include "bruteforce-util.c"

#define CHUNKSIZE 100000

int bruteforce_crack(char *password_hash, char *characters, int password_max_length, int verbose)
{
    int number_of_characters = strlen(characters);
    printf("Brute force of hash: %s\n", password_hash);
    printf("Using %d characters: %s\n", number_of_characters, characters);
    printf("Calculating to a length of %d\n", password_max_length);

    int result = 1;

    static unsigned char buffer[65];
    int i, j, k;

    for (i = 1; i <= password_max_length; i++)
    {
        long possibilities = (long)pow(number_of_characters, i);
        if (verbose)
        {
            printf("Now calculating password length of %d, it has %ld possibilities\n", i, possibilities);
        }

#pragma omp parallel private(buffer, j, k) shared(result)
        {
            char passwordToTest[i + 1];
            for (j = 0; j < possibilities;)
            {
                if (result == 0)
                {
                    printf("Found pass already...\n");
                    //return; //Skip the next processes if we have found the result
                }
                else
                {
                    int nextStep = j + CHUNKSIZE;
// int p = omp_get_thread_num();
// printf("Rank: %d processing\n",p);
#pragma omp for schedule(auto)

                    for (j = j; j < nextStep; j++)
                    {
                        strcpy(passwordToTest, "");
                        int val = j;
                        for (k = 0; k < i; k++)
                        {
                            passwordToTest[k] = characters[val % number_of_characters];
                            val = (int)(val / number_of_characters);
                        }
                        passwordToTest[i] = '\0';
                        hash(passwordToTest, buffer);
                        if (!strcmp(password_hash, buffer))
                        {
#pragma omp critical
                            {
                                printf("Password found: %s\n", passwordToTest);
                                result = 0;
                            }
#pragma omp cancel for
                        }
                    }
                }
            }
        } //End process
    }

    if (result)
    {
        printf("Password not found.\n");
    }
    return result;
}