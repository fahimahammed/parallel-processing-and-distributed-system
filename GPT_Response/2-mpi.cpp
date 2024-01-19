#include <mpi.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <iterator>

using namespace std;

// Comparator for sorting
bool cmp(const pair<string, int> &a, const pair<string, int> &b)
{
    return a.second > b.second;
}

// Function to count words in a string segment
unordered_map<string, int> count_words(const string &str)
{
    istringstream iss(str);
    unordered_map<string, int> counts;
    string word;
    while (iss >> word)
    {
        ++counts[word];
    }
    return counts;
}

int main(int argc, char *argv[])
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start_time = MPI_Wtime();
    unordered_map<string, int> local_word_count;
    string segment;

    // Gather and merge results at root
    if (rank == 0)
    {
        ifstream file("input.txt"); // Replace with your file path
        if (!file.is_open())
        {
            cerr << "Failed to open the file." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        vector<string> lines;
        string line;
        while (getline(file, line))
        {
            lines.push_back(line);
        }
        file.close();

        int count = lines.size();
        for (int i = 1; i < size; ++i)
        {
            int start = i * count / size;
            int end = (i + 1) * count / size;
            int length = end - start;
            MPI_Send(&length, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            for (int j = start; j < end; ++j)
            {
                MPI_Send(lines[j].c_str(), lines[j].size(), MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }

        // The root process also counts words
        unordered_map<string, int> word_count;
        for (int i = 0; i < count / size; ++i)
        {
            auto local_count = count_words(lines[i]);
            for (const auto &pair : local_count)
            {
                word_count[pair.first] += pair.second;
            }
        }

        vector<unordered_map<string, int>> all_counts(size);
        MPI_Gather(&local_word_count, sizeof(unordered_map<string, int>), MPI_BYTE, &all_counts[0], sizeof(unordered_map<string, int>), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Combine word counts from all processes
        unordered_map<string, int> global_word_count;
        for (auto &wc : all_counts)
        {
            for (auto &word : wc)
            {
                global_word_count[word.first] += word.second;
            }
        }

        // Sorting and displaying the results
        vector<pair<string, int>> sorted_words(word_count.begin(), word_count.end());
        sort(sorted_words.begin(), sorted_words.end(), cmp);

        double end_time = MPI_Wtime();
        cout << "Total Time: " << (end_time - start_time) << " seconds" << endl;
        for (int i = 0; i < 10 && i < sorted_words.size(); ++i)
        {
            cout << sorted_words[i].first << ": " << sorted_words[i].second << endl;
        }
    }
    else
    {
        int length;
        MPI_Recv(&length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        char *buffer = new char[length];
        MPI_Recv(buffer, length, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        string str(buffer, length);
        delete[] buffer;
        // Process and send the word count
        auto local_count = count_words(str);
        ostringstream oss;
        for (const auto &pair : local_count)
        {
            oss << pair.first << " " << pair.second << " ";
        }
        string send_str = oss.str();
        MPI_Send(send_str.c_str(), send_str.size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
