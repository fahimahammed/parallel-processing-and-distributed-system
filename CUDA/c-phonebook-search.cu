% % cu
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <sstream>

    using namespace std;

// CUDA kernel for searching a phonebook for a specific name
__global__ void searchPhonebook(const char *data, int totalLength, const char *searchName, int searchNameLength, int *results)
{
    // Thread index and stride calculation for parallel processing
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Iterate through the data in parallel
    for (int i = index; i < totalLength - searchNameLength; i += stride)
    {
        // Check if the current substring matches the search name
        bool found = true;
        for (int j = 0; j < searchNameLength; j++)
        {
            if (data[i + j] != searchName[j])
            {
                found = false;
                break;
            }
        }

        // If a match is found and it is a whole word, increment the results count
        if (found && (i == 0 || data[i - 1] == '\n') && (data[i + searchNameLength] == ' '))
        {
            atomicAdd(results, 1);
        }
    }
}

// Function to read the contents of a file into a string
string readFile(const string &filename)
{
    ifstream file(filename);
    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main()
{
    // Input file containing phonebook data
    string filename = "/content/sample_data/input.txt";

    // Read the phonebook data and define the search name
    string phonebookData = readFile(filename);
    string searchName = "John";

    // CUDA device memory pointers
    char *dData;
    char *dSearchName;
    int *dResults;
    int results = 0;

    // Compute lengths for data and search name
    int dataLength = phonebookData.length();
    int searchNameLength = searchName.length();

    // Allocate device memory and copy data from host to device
    cudaMalloc(&dData, dataLength); // cudaError_t cudaMalloc(void** devPtr, size_t size);
    cudaMalloc(&dSearchName, searchNameLength);
    cudaMalloc(&dResults, sizeof(int));
    cudaMemcpy(dData, phonebookData.c_str(), dataLength, cudaMemcpyHostToDevice); // cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
    cudaMemcpy(dSearchName, searchName.c_str(), searchNameLength, cudaMemcpyHostToDevice);
    cudaMemcpy(dResults, &results, sizeof(int), cudaMemcpyHostToDevice);

    // Configure CUDA kernel execution parameters
    int blockSize = 256;
    int numBlocks = (dataLength + blockSize - 1) / blockSize;

    // CUDA events for measuring execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the CUDA kernel for searching the phonebook
    searchPhonebook<<<numBlocks, blockSize>>>(dData, dataLength, dSearchName, searchNameLength, dResults); // kernel_name<<<grid_size, block_size>>>(arg1, arg2, ..., argN);

    // Record the end time and synchronize events
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the results from device to host and print the results
    cudaMemcpy(&results, dResults, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Total Time: " << milliseconds << " ms" << endl;
    cout << "Number of matching contacts: " << results << endl;

    // Free device memory and destroy CUDA events
    cudaFree(dData);
    cudaFree(dSearchName);
    cudaFree(dResults);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
