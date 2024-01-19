#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <mpi.h>

using namespace std;

double startTime;

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
    int length = text.size();
    sendInt(length, receiver);
    MPI_Send(text.c_str(), length, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

string receiveString(int sender)
{
    int length = receiveInt(sender);
    char *text = new char[length + 1];
    MPI_Recv(text, length, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    text[length] = '\0';
    return string(text);
}

int countPatternOccurrences(const string &text, const string &pattern)
{
    int count = 0;
    size_t pos = text.find(pattern);
    while (pos != string::npos)
    {
        if ((pos > 0 && isalpha(text[pos - 1])) || (pos + pattern.size() < text.size() && isalpha(text[pos + pattern.size()])))
        {
            count++;
        }
        pos = text.find(pattern, pos + 1);
    }
    return count;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    startTime = MPI_Wtime();

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

        string paragraph;
        getline(file, paragraph, '\0'); // Read the whole file as a single string
        file.close();

        int paragraph_size = paragraph.size();

        int segment_size = paragraph_size / (worldSize - 1);
        int remainder = paragraph_size % (worldSize - 1);
        int start = 0;

        for (int i = 1; i < worldSize; i++)
        {
            int segment_end = start + segment_size + (i <= remainder ? 1 : 0);
            string segment = paragraph.substr(start, segment_end - start);
            cout << "sub str : " << segment << endl;
            sendString(segment, i);
            start = segment_end;
        }

        // Master process counts occurrences in its segment
        int masterCount = countPatternOccurrences(paragraph, pattern);

        // Receive and accumulate counts from other processes
        int totalOccurrences = masterCount;
        for (int i = 1; i < worldSize; i++)
        {
            int segmentCount = receiveInt(i);
            totalOccurrences += segmentCount;
        }

        double endTime = MPI_Wtime();
        cout << "Total time: " << endTime - startTime << " seconds" << endl;
        cout << "Number of occurrences of pattern '" << pattern << "': " << totalOccurrences << endl;
    }
    else
    {
        string segment = receiveString(0);
        int segmentCount = countPatternOccurrences(segment, pattern);
        sendInt(segmentCount, 0);
    }

    MPI_Finalize();

    return 0;
}
