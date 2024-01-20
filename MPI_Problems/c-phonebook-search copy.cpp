#include <bits/stdc++.h>
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

void sendString(string text, int receiver)
{
    int length = (int)text.size() + 1;
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

string vectorToString(vector<string> &words, int start, int end)
{
    string text = "";
    for (int i = start; i < min((int)words.size(), end); i++)
    {
        text += words[i] + "\n";
    }
    return text;
}

vector<string> stringToVector(string text)
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

void check(string name, string phone, string searchName, int rank)
{
    if (name.size() != searchName.size())
    {
        return;
    }
    for (int i = 0; i < searchName.size(); i++)
    {
        if (name[i] != searchName[i])
        {
            return;
        }
    }
    double endTime = MPI_Wtime();
    cout << name << " " << phone << " found by process " << rank << "\n";
    cout << "Process " << rank << " took " << endTime - startTime << " seconds\n";
}

void readPhonebook(vector<string> &fileNames, vector<string> &names, vector<string> &phoneNumbers)
{
    for (auto fileName : fileNames)
    {
        ifstream file(fileName);
        string name, number;
        while (file >> name >> number)
        {
            names.push_back(name);
            phoneNumbers.push_back(number);
        }
        file.close();
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    startTime = MPI_Wtime();

    if (!worldRank)
    {
        vector<string> names, phoneNumbers; // name phone list
        vector<string> fileNames(argv + 1, argv + argc);
        readPhonebook(fileNames, names, phoneNumbers);
        int segment = names.size() / worldSize; // change // 5

        for (int i = 1; i < worldSize; i++)
        {
            int start = i * segment, end = start + segment;
            string namesString = vectorToString(names, start, end);
            sendString(namesString, i);
            string phoneNumbersString = vectorToString(phoneNumbers, start, end);
            sendString(phoneNumbersString, i);
        }

        string name = "John";
        for (int i = 0; i < segment; i++)
        {
            check(names[i], phoneNumbers[i], name, worldRank);
        }
    }
    else
    {
        string namesString = receiveString(0);
        vector<string> names = stringToVector(namesString);
        string phoneNumbersString = receiveString(0);
        vector<string> phoneNumbers = stringToVector(phoneNumbersString);

        string name = "John";
        for (int i = 0; i < names.size(); i++)
        {
            check(names[i], phoneNumbers[i], name, worldRank);
        }
    }

    MPI_Finalize();

    return 0;
}
