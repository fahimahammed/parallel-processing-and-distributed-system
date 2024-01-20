#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

const int MAX = 1000000;
int K, M, N, P;

// const int K = 9; //... Total Number of matrices
// const int M = 2; //... Rows of 1st matrix
// const int N = 2; //... Columns of 1st matrix & Rows of 2nd matrix
// const int P = 2; //... Columns of 2nd matrix

// void displayMatrix(int matrix[K][M][N], const char *name, int X)
// {
//     cout << "\n\n"
//          << name << ":\n";
//     for (int k = 0; k < X; k++)
//     {
//         cout << "Matrix " << k << ":\n";
//         for (int r = 0; r < M; r++)
//         {
//             for (int c = 0; c < N; c++)
//             {
//                 cout << matrix[k][r][c] << " ";
//             }
//             cout << "\n";
//         }
//     }
// }

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    double startTime = MPI_Wtime();

    srand(time(nullptr));

    if (argc != 5)
    {
        if (worldRank == 0)
        {
            printf("Usage: %s <K> <M> <N> <P>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    K = atoi(argv[1]);
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    P = atoi(argv[4]);

    // Check input constraints
    if (K * M * N > MAX || K * N * P > MAX || K * M * P > MAX)
    {
        if (worldRank == 0)
        {
            printf("Input dimensions exceed the maximum allowed limit. Please adjust the values.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (K % worldSize != 0)
    {
        if (worldRank == 0)
        {
            cout << "Number of matrices should be divisible by the number of processes\n";
        }
        MPI_Finalize();
        return 0;
    }

    int M1[K][M][N]; //... Array of 1st matrix
    int M2[K][N][P]; //... Array of 2nd matrix
    int M3[K][M][P]; //... Array of result matrix

    if (!worldRank) //... Rank 0 process will create the matrices
    {
        for (int k = 0; k < K; k++)
        {
            for (int r = 0; r < M; r++)
            {
                for (int c = 0; c < N; c++)
                {
                    M1[k][r][c] = rand() % 10;
                }
            }

            for (int r = 0; r < N; r++)
            {
                for (int c = 0; c < P; c++)
                {
                    M2[k][r][c] = rand() % 10;
                }
            }
        }

        // Display matrices M1 and M2
        // displayMatrix(M1, "Matrix M1", 9);
        // displayMatrix(M2, "Matrix M2", 9);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int sizePerProcess = K / worldSize;
    int m1[sizePerProcess][M][N]; //... Local array of 1st matrix
    int m2[sizePerProcess][N][P]; //... Local array of 2nd matrix
    int m3[sizePerProcess][M][P]; //... Local array of result matrix

    MPI_Scatter(M1, sizePerProcess * M * N, MPI_INT, m1, sizePerProcess * M * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(M2, sizePerProcess * N * P, MPI_INT, m2, sizePerProcess * N * P, MPI_INT, 0, MPI_COMM_WORLD);

    // displayMatrix(m1, "Matrix M1", sizePerProcess);
    // displayMatrix(m2, "Matrix M2", sizePerProcess);
    // cout << "\n================>>> Rank: " << worldRank << endl;

    //... Performing matrix multiplication
    for (int n = 0; n < sizePerProcess; n++)
    {
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < P; c++)
            {
                m3[n][r][c] = 0;
                for (int k = 0; k < N; k++)
                {
                    m3[n][r][c] += m1[n][r][k] * m2[n][k][c];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(m3, sizePerProcess * M * P, MPI_INT, M3, sizePerProcess * M * P, MPI_INT, 0, MPI_COMM_WORLD);

    if (!worldRank) //... Rank 0 process will output the result
    {
        for (int k = 0; k < K; k++)
        {
            cout << "\n\n"
                 << "Result " << k << ":\n";
            for (int r = 0; r < M; r++)
            {
                for (int c = 0; c < P; c++)
                {
                    cout << M3[k][r][c] << " ";
                }
                cout << "\n";
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    printf("Process %d took %f seconds.\n", worldRank, MPI_Wtime() - startTime);

    MPI_Finalize();

    return 0;
}
