#include <stdio.h>
#include <mpi.h>
// #include <stdlib.h>

// int arr[] = { 1, 1, 1, 1, 1, 1};
int arr[] = {1, 2, 3, 4};
int n = sizeof(arr) / sizeof(arr[0]);
// Temporary array for slave process
int tempArr[1000] = {0}, tag;

int main(int argc, char *argv[])
{
    int myRank, numOfProcs;

    MPI_Init(&argc, &argv);

    // find out process ID,
    // and how many processes were started
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcs);

    int elementsPerProcess = n / numOfProcs;
    MPI_Status status;

    if (myRank == 0)
    {
        printf("Number of processes: %d\n", numOfProcs);
        printf("Size of array: %d\n", n);
        printf("elementsPerProcess: %d\n", elementsPerProcess);
        int index, i;
        elementsPerProcess = n / numOfProcs;

        // If more than 1 processes are run then distribute task to
        // other processes otherwise all the elements are processed by the master process
        if (numOfProcs > 1)
        {
            // distributes the portion of array to child processes
            for (i = 1; i < numOfProcs - 1; i++)
            {
                index = i * elementsPerProcess;
                tag = 0;
                // send the sub array to child processes
                MPI_Send(&elementsPerProcess, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
                MPI_Send(&arr[index], elementsPerProcess, MPI_INT, i, tag, MPI_COMM_WORLD);
            }

            // last process adds remaining elements
            index = i * elementsPerProcess;
            int elementsLeft = n - index;

            MPI_Send(&elementsLeft, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
            MPI_Send(&arr[index], elementsLeft, MPI_INT, i, tag, MPI_COMM_WORLD);
        }
        // master process adds its own sub array
        int sum = 0;
        // printf("Sum of array is : %d\n", sum);
        for (i = 0; i < elementsPerProcess; i++)
        {
            sum += arr[i];
            // printf("Sum of array is : %d\n", sum);
        }
        // collects partial sums from other processes
        int tmp = 0;
        tag = 1;
        for (i = 1; i < numOfProcs; i++)
        {
            MPI_Recv(&tmp, 1, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
            // int sender = status.MPI_SOURCE;
            sum += tmp;
        }
        // prints the final sum of array
        printf("Sum of array is : %d\n", sum);

    } // end master process
    else
    {
        // slave processes receive sub array from master process
        MPI_Recv(&elementsPerProcess, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&tempArr, elementsPerProcess, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        int partialSum = 0;
        // calculates partial sum of sub array
        for (int i = 0; i < elementsPerProcess; i++)
        {
            partialSum += tempArr[i];
        }
        // sends the partial sum to the master process
        tag = 1;
        MPI_Send(&partialSum, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
    } // end slave process

    // cleans up all MPI state before exit of process
    MPI_Finalize();

    return 0;
}
/*
Sample Input/Output:
Input:
1 2 3 4 5 6 7 8 9 10
Output:
Sum of array is : 55
*/