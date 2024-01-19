#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <string>
#include <chrono>

using namespace std;
using namespace std::chrono;

bool comparePairs(const pair<string, int> &a, const pair<string, int> &b)
{
    return a.second > b.second;
}

string readPartOfFile(ifstream &file, int start, int end)
{
    file.seekg(start);
    string part, line;
    while (file.tellg() < end && getline(file, line))
    {
        part += line + " ";
    }
    return part;
}

void serializeMap(const unordered_map<string, int> &map, vector<string> &keys, vector<int> &values)
{
    for (const auto &pair : map)
    {
        keys.push_back(pair.first);
        values.push_back(pair.second);
    }
}

void deserializeMap(unordered_map<string, int> &map, const vector<string> &keys, const vector<int> &values)
{
    for (size_t i = 0; i < keys.size(); ++i)
    {
        map[keys[i]] += values[i];
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    string filename = "input.txt"; // Replace with your file path
    ifstream file(filename, ios::ate);
    if (!file.is_open())
    {
        cerr << "Failed to open file: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int fileSize = file.tellg();
    int partSize = fileSize / worldSize;
    int start = worldRank * partSize;
    int end = (worldRank == worldSize - 1) ? fileSize : start + partSize;

    string part = readPartOfFile(file, start, end);
    file.close();

    istringstream iss(part);
    unordered_map<string, int> localWordCount;
    string word;
    while (iss >> word)
    {
        ++localWordCount[word];
    }

    vector<string> localKeys;
    vector<int> localValues;
    serializeMap(localWordCount, localKeys, localValues);

    // MPI communication part
    // Assuming that each process has a different number of words to send
    // Gather the sizes of each serialized data first
    int localSize = localKeys.size();
    vector<int> allSizes;
    if (worldRank == 0)
    {
        allSizes.resize(worldSize);
    }
    MPI_Gather(&localSize, 1, MPI_INT, allSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Now gather the keys and values
    vector<string> allKeys;
    vector<int> allValues;
    if (worldRank == 0)
    {
        for (int size : allSizes)
        {
            allKeys.resize(allKeys.size() + size);
            allValues.resize(allValues.size() + size);
        }
    }
    MPI_Gatherv(localKeys.data(), localSize, MPI_CHAR, allKeys.data(), allSizes.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localValues.data(), localSize, MPI_INT, allValues.data(), allSizes.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (worldRank == 0)
    {
        unordered_map<string, int> globalWordCount;
        deserializeMap(globalWordCount, allKeys, allValues);

        vector<pair<string, int>> sortedWords(globalWordCount.begin(), globalWordCount.end());
        sort(sortedWords.begin(), sortedWords.end(), comparePairs);

        auto endTime = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(endTime - startTime).count();
        cout << "Total Time: " << duration << " milliseconds" << endl;

        for (int i = 0; i < 10 && i < sortedWords.size(); ++i)
        {
            cout << sortedWords[i].first << ": " << sortedWords[i].second << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
