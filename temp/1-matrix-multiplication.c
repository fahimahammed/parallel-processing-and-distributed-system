#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

int A[2][2], B[2][2], C[2][2];

MPI_Status s;

int main(int argc, char **argv)
{
    int rank, numOfProcessor, row, i, j, start;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcessor);

    srand(time(NULL));
    if (rank == 0)
    {

        A[0][0] = 1;
        A[0][1] = 2;
        A[1][0] = 2;
        A[1][1] = 3;

        B[0][0] = 2;
        B[0][1] = 2;
        B[1][0] = 4;
        B[1][1] = 3;

        row = 2 / (numOfProcessor - 1);

        start = 0;
        double startTime = MPI_Wtime();
        for (i = 1; i <= numOfProcessor - 1; i++)
        {
            // start, end, row, martrix row, martix b
            MPI_Send(&start, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&row, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&A[start][0], 2 * row, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&B, 2 * 2, MPI_INT, i, 1, MPI_COMM_WORLD);
            start = row + start;
        }

        for (i = 1; i <= numOfProcessor - 1; i++)
        {
            MPI_Recv(&start, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &s);
            MPI_Recv(&row, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &s);
            MPI_Recv(&C[start][0], 2 * row, MPI_INT, i, 2, MPI_COMM_WORLD, &s);
        }

        double endTime = MPI_Wtime();

        printf("\nResult: \n");
        for (i = 0; i < 2; i++)
        {
            for (j = 0; j < 2; j++)
            {
                printf("%d   ", C[i][j]);
            }
            printf("\n");
        }

        printf("\nTime: %f\n", endTime - startTime);
    }

    if (rank > 0)
    {
        MPI_Recv(&start, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &s);
        MPI_Recv(&row, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &s);
        MPI_Recv(&A, 2 * row, MPI_INT, 0, 1, MPI_COMM_WORLD, &s);
        MPI_Recv(&B, 2 * 2, MPI_INT, 0, 1, MPI_COMM_WORLD, &s);

        for (int p = 0; p < 2; p++)
        {
            for (int q = 0; q < row; q++)
            {
                C[p][q] = 0;

                for (int r = 0; r < 2; r++)
                {
                    C[p][q] = C[p][q] + A[q][r] * B[r][p];
                }
            }
        }

        MPI_Send(&start, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&row, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&C, 2 * row, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
