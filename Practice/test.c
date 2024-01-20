#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include <time.h>
#include <sys/time.h>

#define M 2
#define N 2
#define P 2
#define K 2

MPI_Status status;

double matA[K][M][N], matB[K][N][P], matC[K][M][P];

int main(int argc, char **argv)
{
    int size, rank, slaveTaskCount, source, dest, rows, offset;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Validate that the number of processes is appropriate for this program
    if ((size < K || size % K != 0) && rank == 0)
    {
        printf("This program requires a number of processes that is a multiple of %d.\n", K);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Number of slave tasks will be assigned to variable -> slaveTaskCount
    slaveTaskCount = size - 1;

    // Master process
    if (rank == 0)
    {
        // Matrix A and Matrix B both will be filled with random numbers
        srand(time(NULL));
        for (int k = 0; k < K; k++)
        {
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    matA[k][i][j] = rand() % 10;
                }
            }
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < P; j++)
                {
                    matB[k][i][j] = rand() % 10;
                }
            }
        }

        printf("\n\t\tMatrix - Matrix Multiplication using MPI\n");

        // Print Matrix A
        printf("\nMatrix A\n\n");
        for (int k = 0; k < K; k++)
        {
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    printf("%.0f\t", matA[k][i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }

        // Print Matrix B
        printf("\nMatrix B\n\n");
        for (int k = 0; k < K; k++)
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < P; j++)
                {
                    printf("%.0f\t", matB[k][i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }

        // Determine number of rows of the Matrix A, that is sent to each slave process
        rows = M / slaveTaskCount;
        // Offset variable determines the starting point of the row which sent to slave process
        offset = 0;

        // Calculation details are assigned to slave tasks. Process 1 onwards;
        // Each message's tag is 1
        for (dest = 1; dest <= slaveTaskCount; dest++)
        {
            // Acknowledging the offset of the Matrix A
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            // Acknowledging the number of rows
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            // Send rows of the Matrix A which will be assigned to slave process to compute
            MPI_Send(&matA[offset][0][0], rows * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            // Matrix B is sent
            MPI_Send(&matB, K * N * P, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);

            // Offset is modified according to the number of rows sent to each process
            offset = offset + rows;
        }

        // Root process waits until each slave process sends their calculated result with message tag 2
        for (int i = 1; i <= slaveTaskCount; i++)
        {
            source = i;
            // Receive the offset of a particular slave process
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            // Receive the number of rows that each slave process processed
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            // Calculated rows of each process will be stored in Matrix C according to their offset and
            // the processed number of rows
            MPI_Recv(&matC[offset][0][0], rows * P, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }

        // Print the result matrix
        printf("\nResult Matrix C = Matrix A * Matrix B:\n\n");
        for (int k = 0; k < K; k++)
        {
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < P; j++)
                    printf("%.0f\t", matC[k][i][j]);
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }

    // Slave Processes
    if (rank > 0)
    {
        // Source process ID is defined
        source = 0;

        // Slave process waits for the message buffers with tag 1, that Root process sent
        // Each process will receive and execute this separately on their processes

        // The slave process receives the offset value sent by the root process
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        // The slave process receives the number of rows sent by the root process
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        // The slave process receives the sub-portion of Matrix A which assigned by Root
        MPI_Recv(&matA, rows * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        // The slave process receives Matrix B
        MPI_Recv(&matB, K * N * P, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

        // Matrix multiplication
        for (int k = 0; k < K; k++)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < P; j++)
                {
                    // Set the initial value of the row summation
                    matC[k][i][j] = 0.0;
                    // Matrix A's element(i, j) will be multiplied with Matrix B's element(j, k)
                    for (int x = 0; x < N; x++)
                        matC[k][i][j] += matA[k][i][x] * matB[k][x][j];
                }
            }
        }

        // Calculated result will be sent back to Root process (process 0) with message tag 2

        // Offset will be sent to Root, which determines the starting point of the calculated
        // value in matrix C
        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        // Number of rows the process calculated will be sent to root process
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        // Resulting matrix with calculated rows will be sent to root process
        MPI_Send(&matC, K * rows * P, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
