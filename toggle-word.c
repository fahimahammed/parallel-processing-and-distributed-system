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
        // printf("Enter a word: ");
        scanf("%s", word);

        // Send the word to the receiver
        // int MPI_Ssend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
        /**
         * buf: Pointer to the data buffer you want to send.
         * count: Number of elements in the buffer.
         * datatype: MPI data type of the elements in the buffer.
         * dest: Rank of the destination process.
         * tag: Message tag, an integer used to label the message.
         * comm: MPI communicator.
         */
        MPI_Ssend(word, wordSize, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        // Receive the modified word from the receiver
        // int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
        /**
         * buf: Pointer to the data buffer where the received data will be stored.
         * count: Number of elements to be received into the buffer.
         * datatype: MPI data type of the elements in the buffer.
         * source: Rank of the source process (the sender).
         * tag: Message tag, an integer used to label the message.
         * comm: MPI communicator.
         * status: Address of an MPI_Status structure that will hold information about the received message (use MPI_STATUS_IGNORE if you don't need this information).
         */
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
