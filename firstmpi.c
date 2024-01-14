#include <stdio.h>
#include "mpi.h"
#include <string.h>

int main(int argc, char **argv)
{
    int MyRank, Numprocs, tag, ierror, i;
    MPI_Status status;
    char send_message[20], recv_message[20];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
    tag = 100;
    strcpy(send_message, "Hello-Participants");
    if (MyRank == 0)
    {
        for (i = 1; i < Numprocs; i++)
        {
            MPI_Recv(recv_message, 20, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);
            printf("node %d : %s \n", i, recv_message);
        }
    }
    else
        MPI_Send(send_message, 20, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
    MPI_Finalize();
}