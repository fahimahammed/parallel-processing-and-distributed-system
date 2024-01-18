#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MAX_WORD_LENGTH 50
#define TOP_OCCURRENCES 10

// Structure to store word frequency
typedef struct
{
    char word[MAX_WORD_LENGTH];
    int frequency;
} WordFrequency;

// Comparison function for sorting WordFrequency array in descending order
int compareWordFrequency(const void *a, const void *b)
{
    return ((WordFrequency *)b)->frequency - ((WordFrequency *)a)->frequency;
}

// Function to count words in a given string
int countWords(char *str)
{
    int count = 0;
    char delimiters[] = " \t\n\r.,;:!?()-\"";
    char *token = strtok(str, delimiters);

    while (token != NULL)
    {
        count++;
        token = strtok(NULL, delimiters);
    }

    return count;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3)
    {
        if (rank == 0)
        {
            printf("Usage: %s <number_of_processes> <input_file>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int numProcesses = atoi(argv[1]);
    char *inputFile = argv[2];

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // Read the input file
    FILE *file = fopen(inputFile, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file %s\n", inputFile);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Determine the file size
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Read the file content
    char *fileContent = (char *)malloc(fileSize + 1);
    fread(fileContent, 1, fileSize, file);
    fileContent[fileSize] = '\0';

    fclose(file);

    // Split the content for distribution
    int chunkSize = fileSize / numProcesses;
    int *counts = (int *)malloc(numProcesses * sizeof(int));
    int *displs = (int *)malloc(numProcesses * sizeof(int));

    for (int i = 0; i < numProcesses; ++i)
    {
        counts[i] = (i == numProcesses - 1) ? (fileSize - i * chunkSize) : chunkSize;
        displs[i] = i * chunkSize;
    }

    // Broadcast chunk sizes to all processes
    MPI_Bcast(counts, numProcesses, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, numProcesses, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter file content to all processes
    char *localContent = (char *)malloc(counts[rank] + 1);
    MPI_Scatterv(fileContent, counts, displs, MPI_CHAR, localContent, counts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Free memory
    free(fileContent);
    free(counts);
    free(displs);

    // Tokenize local content and count word frequencies
    char *token = strtok(localContent, " \t\n\r");
    WordFrequency *localWordFreq = (WordFrequency *)malloc(sizeof(WordFrequency) * TOP_OCCURRENCES);
    for (int i = 0; i < TOP_OCCURRENCES; ++i)
    {
        localWordFreq[i].frequency = 0;
    }

    while (token != NULL)
    {
        for (int i = 0; i < TOP_OCCURRENCES; ++i)
        {
            if (strcmp(token, localWordFreq[i].word) == 0)
            {
                localWordFreq[i].frequency++;
                break;
            }
            else if (localWordFreq[i].frequency == 0)
            {
                strcpy(localWordFreq[i].word, token);
                localWordFreq[i].frequency = 1;
                break;
            }
        }
        token = strtok(NULL, " \t\n\r");
    }

    // Reduce word frequencies from all processes
    WordFrequency *globalWordFreq = (WordFrequency *)malloc(sizeof(WordFrequency) * TOP_OCCURRENCES * size);
    MPI_Gather(localWordFreq, TOP_OCCURRENCES, MPI_2INT, globalWordFreq, TOP_OCCURRENCES, MPI_2INT, 0, MPI_COMM_WORLD);

    // Rank 0 consolidates results and prints the top occurrences
    if (rank == 0)
    {
        // Combine and sort the results
        qsort(globalWordFreq, TOP_OCCURRENCES * size, sizeof(WordFrequency), compareWordFrequency);

        // Print the top occurrences
        printf("Top %d occurrences:\n", TOP_OCCURRENCES);
        for (int i = 0; i < TOP_OCCURRENCES; ++i)
        {
            printf("%s : %d\n", globalWordFreq[i].word, globalWordFreq[i].frequency);
        }

        // Print the total time taken
        double endTime = MPI_Wtime();
        printf("Total time taken: %.6f seconds\n", endTime - startTime);
    }

    // Free memory
    free(localWordFreq);
    free(globalWordFreq);
    free(localContent);

    MPI_Finalize();

    return 0;
}
