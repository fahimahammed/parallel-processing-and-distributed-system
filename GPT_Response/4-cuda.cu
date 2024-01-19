#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

using namespace std;

__global__ void pattern_count_kernel(const char *text, int length, const char *pattern, int pattern_length, int *count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i <= length - pattern_length; i += stride)
    {
        bool match = true;
        for (int j = 0; j < pattern_length && match; ++j)
        {
            if (pattern[j] != text[i + j])
            {
                match = false;
            }
        }
        if (match)
        {
            atomicAdd(count, 1);
        }
    }
}

string readFile(const string &filename)
{
    ifstream file(filename);
    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main()
{
    string filename = "input.txt"; // Replace with your file path
    string pattern = "%x%";        // Replace with your pattern
    string text = readFile(filename);

    char *d_text;
    char *d_pattern;
    int *d_count;
    int count = 0;

    int text_length = text.length();
    int pattern_length = pattern.length();

    cudaMalloc(&d_text, text_length);
    cudaMalloc(&d_pattern, pattern_length);
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_text, text.c_str(), text_length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, pattern.c_str(), pattern_length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (text_length + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    pattern_count_kernel<<<numBlocks, blockSize>>>(d_text, text_length, d_pattern, pattern_length, d_count);
    cudaEventRecord(stop);

    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Total Time: " << milliseconds << " ms" << endl;
    cout << "Number of occurrences: " << count << endl;

    cudaFree(d_text);
    cudaFree(d_pattern);
    cudaFree(d_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
