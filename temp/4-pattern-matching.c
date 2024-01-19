#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MAX_PATTERN_LENGTH 50
#define MAX_PARAGRAPH_LENGTH 10000

int count_occurrences(const char *paragraph, const char *pattern)
{
    int count = 0;
    const char *ptr = paragraph;

    while ((ptr = strstr(ptr, pattern)) != NULL)
    {
        count++;
        ptr += strlen(pattern);
    }

    return count;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <num_processes> <filename>\n", argv[0]);
        exit(1);
    }

    int num_processes, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int pattern_occurrences = 0;
    double start_time, end_time;

    if (rank == 0)
    {
        // Read paragraph from file
        FILE *file = fopen(argv[2], "r");
        if (file == NULL)
        {
            fprintf(stderr, "Error opening file: %s\n", argv[2]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char paragraph[MAX_PARAGRAPH_LENGTH];
        fgets(paragraph, sizeof(paragraph), file);

        fclose(file);

        // Broadcast the paragraph to all processes
        MPI_Bcast(paragraph, MAX_PARAGRAPH_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Broadcast the pattern to all processes
        char pattern[MAX_PATTERN_LENGTH];
        strcpy(pattern, argv[1]); // Assuming the pattern is given as a command-line argument
        MPI_Bcast(pattern, MAX_PATTERN_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);

        start_time = MPI_Wtime();
    }
    else
    {
        // Receive the paragraph from process 0
        char paragraph[MAX_PARAGRAPH_LENGTH];
        MPI_Bcast(paragraph, MAX_PARAGRAPH_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Receive the pattern from process 0
        char pattern[MAX_PATTERN_LENGTH];
        MPI_Bcast(pattern, MAX_PATTERN_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Calculate the start index and end index for each process
        int paragraph_length = strlen(paragraph);
        int chunk_size = paragraph_length / num_processes;
        int start_index = rank * chunk_size;
        int end_index = (rank == num_processes - 1) ? paragraph_length : start_index + chunk_size;

        // Count occurrences in the assigned portion
        int local_occurrences = count_occurrences(paragraph + start_index, pattern);

        // Sum the local occurrences across all processes
        MPI_Reduce(&local_occurrences, &pattern_occurrences, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        end_time = MPI_Wtime();
    }

    if (rank == 0)
    {
        // Print the results
        printf("Total time: %f seconds\n", end_time - start_time);
        printf("Number of occurrences of the pattern '%s': %d\n", argv[1], pattern_occurrences);
    }

    MPI_Finalize();
    return 0;
}
