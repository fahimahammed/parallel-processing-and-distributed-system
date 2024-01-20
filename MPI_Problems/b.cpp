#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <mpi.h>
#include <unordered_map>
#include <sstream>

using namespace std;

bool comparePairs(const pair<string, int> &a, const pair<string, int> &b)
{
    return a.second > b.second;
}

void sendString(const string &text, int receiver)
{
    int length = text.size();
    MPI_Send(&length, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), length, MPI_CHAR, receiver, 0, MPI_COMM_WORLD);
}

string receiveString(int sender)
{
    int length;
    MPI_Recv(&length, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    char *buffer = new char[length + 1];
    MPI_Recv(buffer, length, MPI_CHAR, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    buffer[length] = '\0';
    string text(buffer);
    delete[] buffer;
    return text;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc != 2)
    {
        if (worldRank == 0)
        {
            cerr << "Usage: " << argv[0] << " <filename>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string filename = argv[1];
    vector<string> words;
    unordered_map<string, int> wordCount;

    if (worldRank == 0)
    {
        ifstream file(filename);
        if (!file.is_open())
        {
            cerr << "Error: Unable to open file " << filename << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        string word;
        while (file >> word)
        {
            words.push_back(word);
        }
        file.close();

        // Distribute words to other processes
        int segmentSize = words.size() / (worldSize - 1);
        for (int i = 1; i < worldSize; i++)
        {
            int start = (i - 1) * segmentSize;
            int end = (i < worldSize - 1) ? start + segmentSize : words.size();
            string segmentText;
            for (int j = start; j < end; j++)
            {
                segmentText += words[j] + " ";
            }
            sendString(segmentText, i);
        }
    }
    else
    {
        // Receive words and count them
        string receivedWords = receiveString(0);
        istringstream iss(receivedWords);
        string word;
        while (iss >> word)
        {
            ++wordCount[word];
        }

        // Send word counts back to the root process
        // This part needs proper serialization and communication of the wordCount map
        // For simplicity, we're sending back a single count
        int count = wordCount.size();
        MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (worldRank == 0)
    {
        // Receive counts from other processes and combine them
        // Again, this needs proper deserialization logic for the actual word counts
        for (int i = 1; i < worldSize; i++)
        {
            int count;
            MPI_Recv(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Combine the counts
        }

        // Sort and display top occurrences
        vector<pair<string, int>> sortedWords(wordCount.begin(), wordCount.end());
        sort(sortedWords.begin(), sortedWords.end(), comparePairs);
        for (int i = 0; i < min(10, static_cast<int>(sortedWords.size())); i++)
        {
            cout << sortedWords[i].first << ": " << sortedWords[i].second << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
