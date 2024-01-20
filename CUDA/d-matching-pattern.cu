% % cu
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

        __global__ void
        countPatternKernel(const char *text, int textLength, const char *pattern, int patternLength, int *count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index <= textLength - patternLength)
    {
        bool found = true;
        for (int i = 0; i < patternLength; i++)
        {
            if (text[index + i] != pattern[i])
            {
                found = false;
                break;
            }
        }
        if (found)
        {
            atomicAdd(count, 1);
        }
    }
}

std::string readFile(const std::string &filename)
{
    std::ifstream file(filename);
    std::string content, line;
    while (getline(file, line))
    {
        content += line + "\n";
    }
    return content;
}

int main()
{
    std::string filename = "/content/sample_data/input.txt";
    std::string text = readFile(filename);
    std::string pattern = "%x%";

    char *d_text, *d_pattern;
    int *d_count;
    int count = 0;

    cudaMalloc(&d_text, text.length());
    cudaMalloc(&d_pattern, pattern.length());
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_text, text.c_str(), text.length(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, pattern.c_str(), pattern.length(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (text.length() + blockSize - 1) / blockSize;
    countPatternKernel<<<numBlocks, blockSize>>>(d_text, text.length(), d_pattern, pattern.length(), d_count);

    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Number of occurrences: " << count << std::endl;

    cudaFree(d_text);
    cudaFree(d_pattern);
    cudaFree(d_count);

    return 0;
}
