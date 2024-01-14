#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int wordSize = 10;
    char word[wordSize];

    if (size != 2)
    {
        fprintf(stderr, "This program requires exactly 2 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0)
    {
        // Sender process
        printf("Enter a word: ");
        scanf("%s", word);

        // Send the word to the receiver
        MPI_Ssend(word, wordSize, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        // Receive the modified word from the receiver
        MPI_Recv(word, wordSize, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Modified word received: %s\n", word);
    }
    else if (rank == 1)
    {
        // Receiver process
        MPI_Recv(word, wordSize, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Toggle each letter in the word
        for (int i = 0; i < wordSize; ++i)
        {
            if (isalpha(word[i]))
            {
                if (islower(word[i]))
                {
                    word[i] = toupper(word[i]);
                }
                else
                {
                    word[i] = tolower(word[i]);
                }
            }
        }

        // Send the modified word back to the sender
        MPI_Ssend(word, wordSize, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
