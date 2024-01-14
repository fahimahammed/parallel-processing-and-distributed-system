#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num1 = 12, num2 = 45;

    if (rank == 1)
    {
        printf("Rank = %d & addition = %d", rank, num1 + num2);
    }
    else if (rank == 2)
    {
        printf("Rank = %d & subtraction  = %d", rank, num1 - num2);
    }
    else if (rank == 3)
    {
        printf("Rank = %d & division  = %d", rank, num1 / num2);
    }
    else if (rank == 4)
    {
        printf("Rank = %d & multiplication   = %d", rank, num1 * num2);
    }
    else
    {
        printf("Rank = %d & I've nothing to do!", rank);
    }

    MPI_Finalize();
    return 0;
}