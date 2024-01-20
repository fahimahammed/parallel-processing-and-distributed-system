#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void multiplyMatrices(double *A, double *B, double *C, int M, int N, int P)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < P; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        printf("Usage: %s <K> <M> <N> <P>\n", argv[0]);
        return 1;
    }

    int K = atoi(argv[1]), M = atoi(argv[2]), N = atoi(argv[3]), P = atoi(argv[4]);
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *A, *B, *C, *localA, *localC;
    double startTime, endTime;

    if (rank == 0)
    {
        A = (double *)malloc(K * M * N * sizeof(double));
        B = (double *)malloc(K * N * P * sizeof(double));
        C = (double *)malloc(K * M * P * sizeof(double));

        // Initialize matrices A and B
        for (int i = 0; i < K * M * N; ++i)
            A[i] = rand() % 10;
        for (int i = 0; i < K * N * P; ++i)
            B[i] = rand() % 10;

        startTime = MPI_Wtime();
    }

    // Distribute matrices A and B among processes
    int rows_per_process = M / size;
    localA = (double *)malloc(rows_per_process * N * sizeof(double));
    localC = (double *)malloc(rows_per_process * P * sizeof(double));

    for (int matrix_index = 0; matrix_index < K; ++matrix_index)
    {
        MPI_Scatter(A + matrix_index * M * N, rows_per_process * N, MPI_DOUBLE,
                    localA, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(B + matrix_index * N * P, N * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        multiplyMatrices(localA, B + matrix_index * N * P, localC, rows_per_process, N, P);

        MPI_Gather(localC, rows_per_process * P, MPI_DOUBLE,
                   C + matrix_index * M * P, rows_per_process * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        endTime = MPI_Wtime();
        printf("Time taken for multiplication: %f seconds\n", endTime - startTime);
        free(A);
        free(B);
        free(C);
    }

    free(localA);
    free(localC);

    MPI_Finalize();
    return 0;
}
