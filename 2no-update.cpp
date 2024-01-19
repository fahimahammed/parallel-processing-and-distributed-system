#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace std::chrono;

bool comparePairs(const pair<string, int> &a, const pair<string, int> &b)
{
    return a.second > b.second; // Sort in descending order of frequency
}

int main()
{
    string filename = "input.txt"; // Replace with your file path
    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Failed to open file: " << filename << endl;
        return 1;
    }

    auto startTime = high_resolution_clock::now();

    unordered_map<string, int> wordCount;
    string word;
    while (file >> word)
    {
        ++wordCount[word];
    }

    file.close();

    vector<pair<string, int>> words(wordCount.begin(), wordCount.end());
    sort(words.begin(), words.end(), comparePairs);

    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(endTime - startTime).count();

    cout << "Total Time: " << duration << " milliseconds" << endl;

    int count = 0;
    for (const auto &pair : words)
    {
        cout << pair.first << ": " << pair.second << endl;
        if (++count == 10)
            break;
    }

    return 0;
}
