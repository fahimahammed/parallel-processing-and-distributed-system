#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include <time.h>
#include <sys/time.h>

#define MAX_DIM 1000

MPI_Status status;

double matA[MAX_DIM][MAX_DIM], matB[MAX_DIM][MAX_DIM], matC[MAX_DIM][MAX_DIM];

int main(int argc, char **argv)
{
    int size, rank, slaveTaskCount, source, dest, rows, offset, K, M, N, P;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    slaveTaskCount = size - 1;

    if (rank == 0)
    {
        printf("Enter the number of matrices (K): ");
        scanf("%d", &K);
        printf("Enter the dimensions M, N, P: ");
        scanf("%d %d %d", &M, &N, &P);

        if (K * M * N > MAX_DIM || K * N * P > MAX_DIM || K * M * P > MAX_DIM)
        {
            printf("Input dimensions exceed the maximum allowed size.\n");
            MPI_Finalize();
            return 1;
        }

        srand(time(NULL));
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                matA[i][j] = rand() % 10;
            }
        }
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < P; j++)
            {
                matB[i][j] = rand() % 10;
            }
        }

        printf("\n\t\tMatrix - Matrix Multiplication using MPI\n");

        printf("\nMatrix A\n\n");
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                printf("%.0f\t", matA[i][j]);
            }
            printf("\n");
        }

        printf("\nMatrix B\n\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < P; j++)
            {
                printf("%.0f\t", matB[i][j]);
            }
            printf("\n");
        }

        rows = M / slaveTaskCount;
        offset = 0;

        for (dest = 1; dest <= slaveTaskCount; dest++)
        {
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&matA[offset][0], rows * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&matB, N * P, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);

            offset = offset + rows;
        }

        for (int i = 1; i <= slaveTaskCount; i++)
        {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&matC[offset][0], rows * P, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }

        printf("\nResult Matrix C = Matrix A * Matrix B:\n\n");
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < P; j++)
                printf("%.0f\t", matC[i][j]);
            printf("\n");
        }
        printf("\n");
    }
    else
    {
        source = 0;
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&matA, rows * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&matB, N * P, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

        for (int k = 0; k < M; k++)
        {
            for (int i = 0; i < rows; i++)
            {
                matC[i][k] = 0.0;
                for (int j = 0; j < N; j++)
                    matC[i][k] = matC[i][k] + matA[i][j] * matB[j][k];
            }
        }

        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&matC, rows * P, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
