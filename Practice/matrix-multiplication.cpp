#include <iostream>
#include <mpi.h>

using namespace std;

const int M = 3;
const int N = 3;
const int P = 3;

void matrix_multiply(int A[M][N], int B[N][P], int C[M][P], int row_start, int row_end)
{
    for (int i = row_start; i < row_end; i++)
    {
        for (int j = 0; j < P; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int A[M][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int B[N][P] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int C[M][P];

    int rows_per_process = M / size;
    int row_start = rank * rows_per_process;
    int row_end = (rank == size - 1) ? M : row_start + rows_per_process;

    matrix_multiply(A, B, C, row_start, row_end);

    int recv_buffer[M][P];
    MPI_Allgather(C, rows_per_process * P, MPI_INT, recv_buffer, rows_per_process * P, MPI_INT, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "Matrix A:" << endl;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << A[i][j] << " ";
            }
            cout << endl;
        }

        cout << "Matrix B:" << endl;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < P; j++)
            {
                cout << B[i][j] << " ";
            }
            cout << endl;
        }

        cout << "Result Matrix C:" << endl;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < P; j++)
            {
                cout << recv_buffer[i][j] << " ";
            }
            cout << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
