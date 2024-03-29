# Parallel Processing & Distributed System Lab

This repository contains parallel and distributed computing programs implemented using MPI (Message Passing Interface) and CUDA (Compute Unified Device Architecture). The programs are designed to explore various parallel processing concepts, including matrix multiplication, word counting, synchronous communication, array addition, and CUDA-based algorithms.

### Lab Tasks
1. Write an MPI program to multiply two matrices of size MxN and NxP
2. Write an MPI program to simulate a simple calculator. Perform each operation using a different process in parallel
3. Write an MPI program to count the words in a file & sort it in descending order of frequency of words that is, the highest occurring word must come first & the least occurring word must come last
4. Write a nMPI program using synchronous send. The sender process sends a word to the receiver. The second process receives the word, toggles each letter of the word and sends it back to the first process. Both processes use synchronous send operations
5. Write an MPI program to add an array of size N using two processes. Print the result in the root process. Investigate the amount of time taken by each process
6. Write a Cuda program for matrix multiplication
7. Write a Cuda program to find out the maximum common subsequence
8. Given a paragraph & a pattern like %x%. Now write a Cuda program to find out the line number where %x% this pattern exists in the given paragraph

### Basic Instructions of MPI

- **To configure the program to run an MPI, initialize all the data structures -**
    ```c
    int MPI_Init(int *argc, char ***argv)
    ```

- **To stop the process & turn off any communication**
    ```c
    int MPI_Finalize()
    ```

- **To get the number of processes -**
    ```c
    int MPI_Comm_size(MPI_Comm comm, int *size)
    ```

- **To get the local processes index -**
    ```c
    int MPI_Comm_rank(MPI_Comm comm, int *rank)
    ```

- **Sending Information -**
  ```c
  int MPI_Send(void *buffer, int elementCount, MPI_Datatype dataType, int destinationRank, int messageTag, MPI_Comm communicator)
  ```
    - `buffer`: Pointer to the data buffer you want to send.
    - `elementCount`: Number of elements in the buffer.
    - `dataType`: MPI data type of the elements in the buffer.
    - `destinationRank`: Rank of the destination process.
    - `messageTag`: Message tag, an integer used to label the message.
    - `communicator`: MPI communicator.

- **Receiving Information -**
  ```c
  int MPI_Recv(void *buffer, int elementCount, MPI_Datatype dataType, int sourceRank, int messageTag, MPI_Comm communicator, MPI_Status *status)
  ```
    - `buffer`: Pointer to the data buffer where the received data will be stored.
    - `elementCount`: Number of elements to be received into the buffer.
    - `dataType`: MPI data type of the elements in the buffer.
    - `sourceRank`: Rank of the source process (the sender).
    - `messageTag`: Message tag, an integer used to label the message.
    - `communicator`: MPI communicator.
    - `status`: Address of an MPI_Status structure that will hold information about the received message (use MPI_STATUS_IGNORE if you don't need this information).

#### Collective Communication
- **Broadcast a Message from One Process to All Other Processes:**
  ```c
  int MPI_Bcast(void *buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
  ```
  - `buf`: Pointer to the data buffer to be broadcast.
  - `count`: Number of elements in the buffer.
  - `datatype`: MPI data type of the elements in the buffer.
  - `root`: Rank of the process broadcasting the message.
  - `comm`: MPI communicator.

- **Scatter Data from One Process to All Processes in a Communicator:**
  ```c
  int MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
  ```
  - `sendbuf`: Pointer to the send buffer.
  - `sendcount`: Number of elements for each process in the send buffer.
  - `sendtype`: MPI data type of the send buffer elements.
  - `recvbuf`: Pointer to the receive buffer.
  - `recvcount`: Number of elements for each process in the receive buffer.
  - `recvtype`: MPI data type of the receive buffer elements.
  - `root`: Rank of the process scattering the data.
  - `comm`: MPI communicator.

- **Gather Data from All Processes in a Communicator to One Process:**
  ```c
  int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
  ```
  - `sendbuf`: Pointer to the send buffer.
  - `sendcount`: Number of elements for each process in the send buffer.
  - `sendtype`: MPI data type of the send buffer elements.
  - `recvbuf`: Pointer to the receive buffer (used by the root process).
  - `recvcount`: Number of elements for each process in the receive buffer.
  - `recvtype`: MPI data type of the receive buffer elements.
  - `root`: Rank of the process gathering the data.
  - `comm`: MPI communicator.

#### Synchronization
- **Synchronize All Processes in a Communicator:**
  ```c
  int MPI_Barrier(MPI_Comm comm)
  ```
  - `comm`: MPI communicator.

#### Reduction Operations
- **Perform a Reduction Operation and Distribute the Result to All Processes:**
  ```c
  int MPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
  ```
  - `sendbuf`: Pointer to the send buffer.
  - `recvbuf`: Pointer to the receive buffer (used by the root process).
  - `count`: Number of elements in the buffer.
  - `datatype`: MPI data type of the elements in the buffer.
  - `op`: MPI operation (e.g., MPI_SUM, MPI_MAX).
  - `root`: Rank of the process performing the reduction operation.
  - `comm`: MPI communicator.
