#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <mpi.h>
#include <bits/stdc++.h>
#include <unordered_map>

using namespace std;

unordered_map<string, int> wordCount;

string serializeMap(const unordered_map<string, int> &map)
{
    stringstream ss;
    for (const auto &pair : map)
    {
        ss << pair.first << ":" << pair.second << ",";
    }
    return ss.str();
}

unordered_map<string, int> deserializeMap(const string &serialized)
{
    unordered_map<string, int> map;
    stringstream ss(serialized);
    string item;
    while (getline(ss, item, ','))
    {
        stringstream itemStream(item);
        string key;
        int value;
        if (getline(itemStream, key, ':'))
        {
            itemStream >> value;
            map[key] = value;
        }
    }
    return map;
}

bool comparePairs(const pair<string, int> &a, const pair<string, int> &b)
{
    return a.second > b.second; // Sort in descending order of frequency
}

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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc != 2)
    {
        if (!worldRank)
        {
            cerr << "Usage: " << argv[0] << " <pattern> <filename>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string filename = argv[1];

    if (!worldRank)
    {
        ifstream file(filename);
        if (!file.is_open())
        {
            cerr << "Error: Unable to open file " << filename << endl;
            MPI_Finalize();
            return 1;
        }

        vector<string> words;
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
            int start = (i - 1) * segmentSize + min(i - 1, remainingSegmentSize);
            int end = start + segmentSize + (i <= remainingSegmentSize ? 1 : 0);
            string segmentString = vectorToString(words, start, end);
            sendString(segmentString, i);
        }

        for (int i = 1; i < worldSize; i++)
        {
            // Receive the length of the serialized string
            int length;
            MPI_Recv(&length, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Allocate buffer and receive the serialized string
            char *buffer = new char[length + 1];
            MPI_Recv(buffer, length, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            buffer[length] = '\0'; // Null-terminate the string

            // Deserialize the string back into a map
            unordered_map<string, int> receivedMap = deserializeMap(string(buffer));
            delete[] buffer;

            // Combine receivedMap into the main wordCount map
            for (const auto &pair : receivedMap)
            {
                wordCount[pair.first] += pair.second;
                cout << "=== " << wordCount[pair.first] << " : " << pair.second << endl;
            }
        }

        vector<pair<string, int>> wordsWorld(wordCount.begin(), wordCount.end());
        sort(wordsWorld.begin(), wordsWorld.end(), comparePairs);

        int count = 0;
        for (const auto &pair : wordsWorld)
        {
            cout << pair.first << ": " << pair.second << endl;
            if (++count == 10)
                break;
        }
    }
    else
    {
        string segmentString = receiveString(0);
        vector<string> segmentVector = stringToVector(segmentString);
        for (const string &word : segmentVector)
        {
            ++wordCount[word];
        }

        string serializedMap = serializeMap(wordCount);
        int length = serializedMap.size();
        MPI_Send(&length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        // Then send the serialized string
        MPI_Send(serializedMap.c_str(), length, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
