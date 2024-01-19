#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

using namespace std;

// Function to compare pairs (for sorting)
bool comparePairs(const pair<string, int> &a, const pair<string, int> &b)
{
    return a.second > b.second; // sort in descending order of frequency
}

int main()
{
    string filename = "input.txt"; // Replace with your file path
    ifstream file(filename);
    unordered_map<string, int> wordCount;

    if (!file.is_open())
    {
        cerr << "Failed to open file: " << filename << endl;
        return 1;
    }

    // Read words from the file and count their occurrences
    string word;
    while (file >> word)
    {
        ++wordCount[word];
    }

    file.close();

    // Transfer to a vector for sorting
    vector<pair<string, int>> words;
    for (const auto &pair : wordCount)
    {
        words.push_back(pair);
    }

    // Sort the vector by frequency
    sort(words.begin(), words.end(), comparePairs);

    // Display the words and their frequencies
    for (const auto &pair : words)
    {
        cout << pair.first << ": " << pair.second << endl;
    }

    return 0;
}
