#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <mpi.h>
#include <bits/stdc++.h>

using namespace std;

vector<string> words;

void sendInt(int number, int receiver)
{
    MPI_Send(&number, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
}

int receiveInt(int sender)
{
    int number;
    MPI_Recv(&number, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return number;
}

void sendString(const string &text, int receiver)
{
    int length = static_cast<int>(text.size()) + 1;
    sendInt(length, receiver);
    MPI_Send(&text[0], length, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

string receiveString(int sender)
{
    int length = receiveInt(sender);
    char *text = new char[length];
    MPI_Recv(text, length, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return string(text);
}

string vectorToString(const vector<string> &words, int start, int end)
{
    string text = "";
    for (int i = start; i < min(static_cast<int>(words.size()), end); i++)
    {
        text += words[i] + "\n";
    }
    return text;
}

vector<string> stringToVector(const string &text)
{
    stringstream x(text);
    vector<string> words;
    string word;
    while (x >> word)
    {
        words.push_back(word);
    }
    return words;
}

int countPatternOccurrences(const vector<string> &words, const string &pattern)
{
    int occurrences = 0;
    for (const auto &w : words)
    {
        if (pattern.size() == 3 && pattern[0] == '%' && pattern[2] == '%' &&
            w.find(pattern[1]) != string::npos && w.find(pattern[1]) != 0 &&
            w.find(pattern[1]) != w.size() - 1)
        {
            occurrences++;
        }
    }
    return occurrences;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc != 3)
    {
        if (!worldRank)
        {
            cerr << "Usage: " << argv[0] << " <pattern> <filename>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string pattern = argv[1];
    string filename = argv[2];

    if (!worldRank)
    {
        ifstream file(filename);
        if (!file.is_open())
        {
            cerr << "Error: Unable to open file " << filename << endl;
            MPI_Finalize();
            return 1;
        }

        string word;
        while (file >> word)
        {
            words.push_back(word);
        }
        file.close();

        int totalWords = words.size();
        int segmentSize = totalWords / (worldSize - 1);
        int remainingSegmentSize = totalWords % (worldSize - 1);

        for (int i = 1; i < worldSize; i++)
        {
            int start = i * segmentSize, end = start + segmentSize;
            string segmentString = vectorToString(words, start, end);
            sendString(segmentString, i);
        }

        vector<string> segment;
        for (int i = 0; i < segmentSize; i++)
        {
            segment.push_back(words[i]);
        }

        int totalOccurrences = countPatternOccurrences(segment, pattern);
        for (int i = 1; i < worldSize; i++)
        {
            totalOccurrences += receiveInt(i);
        }

        cout << "===> occurrences " << totalOccurrences << endl;
    }
    else
    {
        string segmentString = receiveString(0);
        vector<string> segmentVector = stringToVector(segmentString);
        int occurrences = countPatternOccurrences(segmentVector, pattern);
        sendInt(occurrences, 0);
    }
    MPI_Finalize();

    return 0;
}
